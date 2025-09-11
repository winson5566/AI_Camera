import os
import time
import json
import random
import glob
import numpy as np
from PIL import Image
# 适配树莓派：优先使用 tflite-runtime
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ----------------- 配置 -----------------
MODEL_PATH = "model/model_efficientnet_b0_inat2021_drq.tflite"
CATEGORIES_JSON = "inat2021/categories.json"
TEST_DIR = "test"
TOP_K = 1  # 只取Top-1
CENTER_CROP = True
# ---------------------------------------

# ---------- 工具函数 ----------
def load_categories(categories_json):
    """加载分类文件，返回 index->name 映射"""
    with open(categories_json, 'r', encoding='utf-8') as f:
        cats = json.load(f)
    idx_to_name = {}
    for i, c in enumerate(cats):
        idx = int(c.get('id', i))
        name = c.get('common_name') or c.get('name') or f"class_{idx}"
        idx_to_name[idx] = name
    return idx_to_name

def load_and_preprocess_image(path, size, center_crop=True):
    """加载图片并预处理为模型输入格式"""
    img = Image.open(path).convert('RGB')
    if center_crop:
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
    img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(img)
    return arr

def fix_to_uint8(x_np: np.ndarray) -> np.ndarray:
    """确保图片数据为uint8格式"""
    if x_np.dtype.kind == 'f':
        if float(x_np.max()) <= 1.0 and float(x_np.min()) >= 0.0:
            x_np = np.round(x_np * 255.0)
        x_np = np.clip(x_np, 0.0, 255.0).astype(np.uint8, copy=False)
    else:
        x_np = np.clip(x_np, 0, 255).astype(np.uint8, copy=False)
    return x_np

def prepare_input(interpreter, x_np: np.ndarray):
    """将预处理后的图片设置到TFLite模型输入"""
    input_details = interpreter.get_input_details()[0]
    in_index = input_details['index']
    in_dtype = input_details['dtype']  # 模型输入类型
    wanted_shape = list(input_details['shape'])

    x_u8 = fix_to_uint8(x_np)  # 保证是uint8
    target_shape = [1, x_u8.shape[0], x_u8.shape[1], x_u8.shape[2]]

    # 如果模型的输入shape与图片不一致，动态调整
    if wanted_shape != target_shape:
        interpreter.resize_tensor_input(in_index, target_shape, strict=False)
        interpreter.allocate_tensors()

    # 根据输入类型进行处理
    if in_dtype == np.uint8:
        # 模型直接接受uint8
        x_for_model = x_u8[None, ...]
    elif in_dtype == np.int8:  # INT8 模型
        scale, zero_point = input_details.get('quantization', (None, None))
        if scale in (None, 0.0):
            x_q = x_u8.astype(np.int32) - 128
            x_q = np.clip(x_q, -128, 127).astype(np.int8, copy=False)
        else:
            x_q = np.round(x_u8.astype(np.float32) / float(scale) + float(zero_point))
            x_q = np.clip(x_q, -128, 127).astype(np.int8, copy=False)
        x_for_model = x_q[None, ...]
    elif in_dtype == np.float32:  # Float32 模型
        # 直接转float32，不做归一化
        x_for_model = x_u8.astype(np.float32, copy=False)[None, ...]
        # 如果需要归一化到0~1，使用下面这行
        # x_for_model = (x_u8.astype(np.float32) / 255.0)[None, ...]
    else:
        raise ValueError(f"Unsupported input dtype: {in_dtype}")

    interpreter.set_tensor(in_index, x_for_model)


def maybe_dequantize_output(output_details, y):
    """输出反量化"""
    y = y.astype(np.float32, copy=False)
    scale, zero_point = output_details.get('quantization', (None, None))
    if scale not in (None, 0.0):
        y = scale * (y - float(zero_point))
    return y

# ---------- 主推理函数 ----------
def run_inference_on_image(interpreter, input_size, output_details, img_path):
    """对单张图片进行推理"""
    arr = load_and_preprocess_image(img_path, size=input_size, center_crop=CENTER_CROP)
    prepare_input(interpreter, arr)

    # 计时
    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()
    infer_ms = (t1 - t0) * 1000.0

    # 获取输出
    y = interpreter.get_tensor(output_details['index'])
    if y.ndim == 2 and y.shape[0] == 1:
        y = y[0]
    y = maybe_dequantize_output(output_details, y)

    # Top-1
    cls = int(np.argmax(y))
    score = float(y[cls])
    return infer_ms, cls, score

def main():
    # 1. 加载类别
    idx_to_name = load_categories(CATEGORIES_JSON)
    print(f"[INFO] Loaded {len(idx_to_name)} categories.")

    # 2. 加载TFLite模型
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_size = int(interpreter.get_input_details()[0]['shape'][1])
    output_details = interpreter.get_output_details()[0]
    print(f"[INFO] Loaded TFLite model: {MODEL_PATH}")
    print(f"[INFO] Input size: {input_size}x{input_size}")

    # 3. 获取测试图片列表
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(TEST_DIR, ext)))
    if not img_paths:
        print("[ERROR] No test images found in 'test/' directory.")
        return

    print(f"[INFO] Found {len(img_paths)} test images, running inference...\n")

    # 4. 遍历推理
    for img_path in img_paths:
        infer_ms, cls, score = run_inference_on_image(interpreter, input_size, output_details, img_path)
        pred_name = idx_to_name.get(cls, f"Class {cls}")
        print(f"[{os.path.basename(img_path)}] -> {pred_name} "
              f"(Score: {score*100:.2f}%, Time: {infer_ms:.2f} ms)")

if __name__ == "__main__":
    main()
