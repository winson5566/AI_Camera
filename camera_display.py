# -*- coding:utf-8 -*-
import time
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import ST7789
from picamera2 import Picamera2
import json
import os
import textwrap
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

# 休眠超时时间（秒）
SLEEP_TIMEOUT = 60
DEFAULT_BRIGHTNESS = 50  # 正常工作时背光亮度 (0-100)

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
    """保存图片到history文件夹 (保持原始未旋转方向)"""
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
disp.bl_DutyCycle(DEFAULT_BRIGHTNESS)
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
MODE_DELETE_CONFIRM = 3
MODE_SLEEP = 4  # 新增休眠模式

mode = MODE_PREVIEW
captured_image = None
gallery_index = 0
gallery_files = []
delete_selection = 0
sleeping = False  # 是否处于休眠状态

# 上次按键时间
last_active = time.time()

# ---------- 按键检测函数 ----------
def check_any_key():
    """检测是否有任意按键被按下"""
    return (
        disp.digital_read(disp.GPIO_KEY_PRESS_PIN) == 1 or
        disp.digital_read(disp.GPIO_KEY1_PIN) == 1 or
        disp.digital_read(disp.GPIO_KEY2_PIN) == 1 or
        disp.digital_read(disp.GPIO_KEY3_PIN) == 1 or
        disp.digital_read(disp.GPIO_KEY_UP_PIN) == 1 or
        disp.digital_read(disp.GPIO_KEY_DOWN_PIN) == 1 or
        disp.digital_read(disp.GPIO_KEY_LEFT_PIN) == 1 or
        disp.digital_read(disp.GPIO_KEY_RIGHT_PIN) == 1
    )

# ---------- 进入休眠 ----------
def enter_sleep():
    global sleeping, mode
    picam2.stop()
    disp.bl_DutyCycle(0)  # 关闭背光
    sleeping = True
    mode = MODE_SLEEP
    logging.info("进入休眠模式")

# ---------- 唤醒 ----------
def wake_up():
    global sleeping, mode
    picam2.start()
    disp.bl_DutyCycle(DEFAULT_BRIGHTNESS)
    sleeping = False
    mode = MODE_PREVIEW
    logging.info("唤醒设备")

# ---------- 主循环 ----------
try:
    while True:
        now = time.time()

        # 检测是否有按键输入
        if check_any_key():
            last_active = now
            if sleeping:
                wake_up()
                time.sleep(0.3)  # 防止误触发
                continue

        # 超时进入休眠
        if not sleeping and (now - last_active > SLEEP_TIMEOUT):
            enter_sleep()

        # 如果处于休眠状态，则跳过其他逻辑
        if sleeping:
            time.sleep(0.2)
            continue

        # ---------- 不同模式的显示 ----------
        if mode == MODE_PREVIEW:
            frame = picam2.capture_array()
            img_pil = Image.fromarray(frame)
            disp.ShowImage(img_pil.rotate(0))

        elif mode == MODE_RESULT:
            if captured_image:
                disp.ShowImage(captured_image)

        elif mode == MODE_GALLERY:
            if gallery_files:
                img_path = gallery_files[gallery_index]
                base_img = Image.open(img_path).resize((240, 240))
                draw = ImageDraw.Draw(base_img)
                font = ImageFont.truetype(FONT_PATH, 16)
                index_text = f"({gallery_index + 1}/{len(gallery_files)})"
                draw.rectangle((0, 0, 240, 20), fill=(0, 0, 0))
                draw.text((10, 2), index_text, font=font, fill=(255, 255, 255))
                disp.ShowImage(base_img.rotate(0))

        elif mode == MODE_DELETE_CONFIRM:
            confirm_img = Image.new("RGB", (240, 240), (0, 0, 0))
            draw = ImageDraw.Draw(confirm_img)
            font = ImageFont.truetype(FONT_PATH, 20)
            options = ["DELETE", "CANCEL"]
            for i, opt in enumerate(options):
                color = (255, 0, 0) if i == delete_selection else (255, 255, 255)
                draw.text((80, 100 + i * 40), opt, font=font, fill=color)
            disp.ShowImage(confirm_img.rotate(0))

        # ---------- 按键变量 ----------
        center_pressed = disp.digital_read(disp.GPIO_KEY_PRESS_PIN) == 1 or disp.digital_read(disp.GPIO_KEY1_PIN) == 1
        key_gallery = disp.digital_read(disp.GPIO_KEY2_PIN) == 1
        key_delete = disp.digital_read(disp.GPIO_KEY3_PIN) == 1
        key_up = disp.digital_read(disp.GPIO_KEY_UP_PIN) == 1
        key_down = disp.digital_read(disp.GPIO_KEY_DOWN_PIN) == 1
        key_left = disp.digital_read(disp.GPIO_KEY_LEFT_PIN) == 1
        key_right = disp.digital_read(disp.GPIO_KEY_RIGHT_PIN) == 1

        # ====== 拍照推理 ======
        if center_pressed and mode == MODE_PREVIEW:
            time.sleep(0.2)
            infer_ms, cls, score = run_inference(img_pil)
            pred_name = idx_to_name.get(cls, f"Class {cls}")

            text = f"{pred_name} ({score * 100:.1f}%)"
            wrapped = textwrap.wrap(text, width=18)[:2]

            # 1. 保存原始照片，不旋转
            save_history_image(img_pil)

            # 2. 创建透明的文字层
            overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # 3. 在 overlay 上绘制黑色背景条和文字
            font = ImageFont.truetype(FONT_PATH, 18)
            draw.rectangle((0, 200, 240, 240), fill=(0, 0, 0, 255))
            for i, line in enumerate(wrapped):
                draw.text((10, 200 + i * 20), line, font=font, fill=(255, 255, 255, 255))

            # 4. 只旋转 overlay，不旋转原始照片
            overlay = overlay.transpose(Image.ROTATE_270)

            # 5. 将 overlay 叠加到照片上
            img_with_text = Image.alpha_composite(img_pil.convert("RGBA"), overlay)

            # 6. 用于显示的最终结果
            captured_image = img_with_text.convert("RGB")
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
                gallery_index = len(gallery_files) - 1
                mode = MODE_GALLERY
                logging.info("进入相册模式")

        # ====== 相册浏览 ======
        elif mode == MODE_GALLERY:
            if key_up or key_left:
                time.sleep(0.2)
                gallery_index = (gallery_index - 1) % len(gallery_files)
            elif key_down or key_right:
                time.sleep(0.2)
                gallery_index = (gallery_index + 1) % len(gallery_files)
            elif key_delete:
                time.sleep(0.2)
                mode = MODE_DELETE_CONFIRM
                delete_selection = 0
            elif center_pressed:
                time.sleep(0.2)
                mode = MODE_PREVIEW

        # ====== 删除确认处理 ======
        elif mode == MODE_DELETE_CONFIRM:
            if key_up or key_down:
                delete_selection = 1 - delete_selection
                time.sleep(0.2)
            elif center_pressed:
                if delete_selection == 0:
                    os.remove(gallery_files[gallery_index])
                    del gallery_files[gallery_index]
                    if gallery_files:
                        gallery_index %= len(gallery_files)
                        mode = MODE_GALLERY
                    else:
                        mode = MODE_PREVIEW
                    logging.info("图片已删除")
                else:
                    mode = MODE_GALLERY
                time.sleep(0.2)

except KeyboardInterrupt:
    logging.info("用户手动退出程序")

finally:
    picam2.stop()
    disp.clear()
    disp.module_exit()
    logging.info("程序已退出，资源释放完成")
