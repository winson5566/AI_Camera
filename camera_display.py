# -*- coding:utf-8 -*-
import time
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import ST7789
from picamera2 import Picamera2
import json
import os
from datetime import datetime

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ----------------- 配置 -----------------
MODEL_PATH = "model/model_efficientnet_b0_inat2021_drq.tflite"
CATEGORIES_JSON = "inat2021/categories.json"
HISTORY_DIR = "/home/winson/AI_Camera/history"  # 历史照片存储路径
TOP_K = 1
CENTER_CROP = True
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
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

# ---------- 工具函数 ----------
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
    if CENTER_CROP:
        w, h = img_pil.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img_pil = img_pil.crop((left, top, left + s, top + s))

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

def save_history_image(img_pil):
    """保存图片到history文件夹"""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.jpg"
    path = os.path.join(HISTORY_DIR, filename)
    img_pil.save(path, "JPEG")
    logging.info(f"保存照片到: {path}")
    return path

def load_history_images():
    """加载history文件夹中的图片列表"""
    if not os.path.exists(HISTORY_DIR):
        return []
    files = sorted([f for f in os.listdir(HISTORY_DIR) if f.lower().endswith(".jpg")])
    return [os.path.join(HISTORY_DIR, f) for f in files]

# ---------- 初始化 ST7789 ----------
disp = ST7789.ST7789()
disp.Init()
disp.clear()
disp.bl_DutyCycle(100)
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
MODE_PREVIEW = 0
MODE_RESULT = 1
MODE_GALLERY = 2

mode = MODE_PREVIEW
captured_image = None
gallery_index = 0
gallery_files = []

# ---------- 主循环 ----------
try:
    while True:
        if mode == MODE_PREVIEW:
            # 实时预览
            frame = picam2.capture_array()
            img_pil = Image.fromarray(frame).rotate(270)
            disp.ShowImage(img_pil)

        elif mode == MODE_RESULT:
            # 显示拍照结果
            if captured_image:
                disp.ShowImage(captured_image)

        elif mode == MODE_GALLERY:
            if gallery_files:
                img_path = gallery_files[gallery_index]
                img = Image.open(img_path).resize((240, 240))
                disp.ShowImage(img)

        # ---------- 按键检测 ----------
        center_pressed = (
            disp.digital_read(disp.GPIO_KEY_PRESS_PIN) == 1 or
            disp.digital_read(disp.GPIO_KEY1_PIN) == 1
        )
        key_gallery = disp.digital_read(disp.GPIO_KEY2_PIN) == 1
        key_exit_gallery = disp.digital_read(disp.GPIO_KEY3_PIN) == 1

        key_up = disp.digital_read(disp.GPIO_KEY_UP_PIN) == 1
        key_down = disp.digital_read(disp.GPIO_KEY_DOWN_PIN) == 1
        key_left = disp.digital_read(disp.GPIO_KEY_LEFT_PIN) == 1
        key_right = disp.digital_read(disp.GPIO_KEY_RIGHT_PIN) == 1

        # ====== 从预览进入拍照推理 ======
        if center_pressed and mode == MODE_PREVIEW:
            time.sleep(0.2)
            infer_ms, cls, score = run_inference(img_pil)
            pred_name = idx_to_name.get(cls, f"Class {cls}")

            # 显示推理结果
            img_draw = img_pil.rotate(-270)
            draw = ImageDraw.Draw(img_draw)
            font = ImageFont.truetype(FONT_PATH, 18)
            text = f"{pred_name} ({score * 100:.1f}%)"
            draw.rectangle((0, 200, 240, 240), fill=(0, 0, 0))
            draw.text((10, 210), text, font=font, fill=(255, 255, 255))
            img_final = img_draw.rotate(270)

            # 保存图片
            save_history_image(img_final)

            captured_image = img_final.copy()
            mode = MODE_RESULT
            logging.info(f"推理结果: {text} | Time: {infer_ms:.2f} ms")

        # ====== 从拍照结果返回预览 ======
        elif center_pressed and mode == MODE_RESULT:
            time.sleep(0.2)
            mode = MODE_PREVIEW

        # ====== 进入相册模式 ======
        elif key_gallery and mode != MODE_GALLERY:
            time.sleep(0.2)
            gallery_files = load_history_images()
            if gallery_files:
                gallery_index = len(gallery_files) - 1  # 默认显示最新一张
                mode = MODE_GALLERY
                logging.info("进入相册模式")

        # ====== 相册中切换图片 ======
        elif mode == MODE_GALLERY:
            if key_up or key_left:
                time.sleep(0.2)
                if gallery_files:
                    gallery_index = (gallery_index - 1) % len(gallery_files)
                    logging.info(f"上一张: {gallery_index + 1}/{len(gallery_files)}")

            elif key_down or key_right:
                time.sleep(0.2)
                if gallery_files:
                    gallery_index = (gallery_index + 1) % len(gallery_files)
                    logging.info(f"下一张: {gallery_index + 1}/{len(gallery_files)}")

            # ====== 退出相册 ======
            if key_exit_gallery or center_pressed:
                time.sleep(0.2)
                mode = MODE_PREVIEW
                logging.info("退出相册模式")

except KeyboardInterrupt:
    logging.info("用户手动退出程序")

finally:
    picam2.stop()
    disp.clear()
    disp.module_exit()
    logging.info("程序已退出，资源释放完成")
