# -*- coding:utf-8 -*-
import time
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import ST7789
from picamera2 import Picamera2

# ---------- 模型相关 ----------
import json
import os
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ----------------- 配置 -----------------
MODEL_PATH = "model/model_efficientnet_b0_inat2021_drq.tflite"
CATEGORIES_JSON = "inat2021/categories.json"
TOP_K = 1
CENTER_CROP = True
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # 树莓派默认字体路径
# ---------------------------------------

logging.basicConfig(level=logging.INFO)

# ---------- 加载类别 ----------
def load_categories(categories_json):
    with open(categories_json, 'r', encoding='utf-8') as f:
        cats = json.load(f)
    idx_to_name = {}
    for i, c in enumerate(cats):
        idx = int(c.get('id', i))
        name = c.get('common_name') or c.get('name') or f"class_{idx}"
        idx_to_name[idx] = name
    return idx_to_name

# ---------- 模型初始化 ----------
logging.info("Loading TFLite model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_size = int(interpreter.get_input_details()[0]['shape'][1])
output_details = interpreter.get_output_details()[0]
idx_to_name = load_categories(CATEGORIES_JSON)
logging.info(f"Model loaded. Input size: {input_size}x{input_size}, {len(idx_to_name)} classes")

def fix_to_uint8(x_np: np.ndarray) -> np.ndarray:
    """确保图片数据为uint8格式"""
    if x_np.dtype.kind == 'f':
        if float(x_np.max()) <= 1.0 and float(x_np.min()) >= 0.0:
            x_np = np.round(x_np * 255.0)
        x_np = np.clip(x_np, 0.0, 255.0).astype(np.uint8, copy=False)
    else:
        x_np = np.clip(x_np, 0, 255).astype(np.uint8, copy=False)
    return x_np

def prepare_input(interpreter, img_pil: Image.Image):
    """将拍照得到的PIL图像设置到模型输入"""
    # 中心裁剪
    if CENTER_CROP:
        w, h = img_pil.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img_pil = img_pil.crop((left, top, left + s, top + s))

    # 调整大小
    img_pil = img_pil.resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(img_pil)
    arr = fix_to_uint8(arr)

    input_details = interpreter.get_input_details()[0]
    in_index = input_details['index']
    in_dtype = input_details['dtype']

    if in_dtype == np.uint8:
        x_for_model = arr[None, ...]
    elif in_dtype == np.int8:
        scale, zero_point = input_details.get('quantization', (None, None))
        if scale in (None, 0.0):
            x_q = arr.astype(np.int32) - 128
            x_q = np.clip(x_q, -128, 127).astype(np.int8, copy=False)
        else:
            x_q = np.round(arr.astype(np.float32) / float(scale) + float(zero_point))
            x_q = np.clip(x_q, -128, 127).astype(np.int8, copy=False)
        x_for_model = x_q[None, ...]
    elif in_dtype == np.float32:
        x_for_model = arr.astype(np.float32, copy=False)[None, ...]
    else:
        raise ValueError(f"Unsupported input dtype: {in_dtype}")

    interpreter.set_tensor(in_index, x_for_model)

def maybe_dequantize_output(y):
    """输出反量化"""
    y = y.astype(np.float32, copy=False)
    scale, zero_point = output_details.get('quantization', (None, None))
    if scale not in (None, 0.0):
        y = scale * (y - float(zero_point))
    return y

def run_inference(img_pil):
    """对拍照的PIL图像进行推理"""
    prepare_input(interpreter, img_pil)
    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()
    infer_ms = (t1 - t0) * 1000.0

    y = interpreter.get_tensor(output_details['index'])
    if y.ndim == 2 and y.shape[0] == 1:
        y = y[0]
    y = maybe_dequantize_output(y)

    cls = int(np.argmax(y))
    score = float(y[cls])
    return infer_ms, cls, score

# ---------- 初始化 ST7789 ----------
disp = ST7789.ST7789()
disp.Init()
disp.clear()
disp.bl_DutyCycle(100)  # 背光亮度
logging.info("ST7789 初始化完成")

# ---------- 初始化摄像头 ----------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (240, 240), "format": "BGR888"}
)
picam2.configure(config)
picam2.start()
logging.info("Picamera2 初始化完成")

# ---------- 模式状态 ----------
preview_mode = True
captured_image = None

try:
    while True:
        if preview_mode:
            # 实时预览
            frame = picam2.capture_array()
            img_pil = Image.fromarray(frame).rotate(270)
            disp.ShowImage(img_pil)

        else:
            if captured_image:
                disp.ShowImage(captured_image)

        # 检测 Center 按键
        center_pressed = (disp.digital_read(disp.GPIO_KEY_PRESS_PIN) == 1)
        if center_pressed:
            time.sleep(0.2)  # 按键防抖

            if preview_mode:
                # ======== 拍照并推理 ========
                infer_ms, cls, score = run_inference(img_pil)
                pred_name = idx_to_name.get(cls, f"Class {cls}")

                # 先旋转回原始方向绘制文字
                img_draw = img_pil.rotate(-270)  # 或 rotate(90)
                draw = ImageDraw.Draw(img_draw)
                font = ImageFont.truetype(FONT_PATH, 18)
                text = f"{pred_name} ({score * 100:.1f}%)"

                # 在底部绘制黑条和文字
                draw.rectangle((0, 200, 240, 240), fill=(0, 0, 0))
                draw.text((10, 210), text, font=font, fill=(255, 255, 255))

                # 再旋转回来
                img_pil = img_draw.rotate(270)

                captured_image = img_pil.copy()
                preview_mode = False
                logging.info(f"推理结果: {text} | Time: {infer_ms:.2f} ms")
            else:
                # 返回实时预览模式
                preview_mode = True

except KeyboardInterrupt:
    logging.info("用户手动退出程序")

finally:
    picam2.stop()
    disp.clear()
    disp.module_exit()
    logging.info("程序已退出，资源释放完成")
