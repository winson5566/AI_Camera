# -*- coding: utf-8 -*-
import os
import time
import json
import random
import glob
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 优先使用 tflite-runtime（树莓派）
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ----------------- 配置 -----------------
MODEL_PATH_DEFAULT = "model/model_efficientnet_b0_inat2021_drq.tflite"
CATEGORIES_JSON = "inat2021/categories.json"
TEST_DIR = "test"
CENTER_CROP = True

INTERVAL_SEC = 10.0   # 每 10 秒一次推理

# 显示屏（可选）
USE_DISPLAY = True
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 18
DEFAULT_BRIGHTNESS = 100
# ---------------------------------------


# ---------- 工具函数 ----------
def load_categories(categories_json):
    with open(categories_json, "r", encoding="utf-8") as f:
        cats = json.load(f)
    return {
        int(c.get("id", i)): c.get("common_name") or c.get("name") or f"class_{i}"
        for i, c in enumerate(cats)
    }


def load_and_preprocess_image(path, size):
    img = Image.open(path).convert("RGB")
    if CENTER_CROP:
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s)//2, (h - s)//2, (w + s)//2, (h + s)//2))
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def prepare_input(interpreter, x_np):
    inp = interpreter.get_input_details()[0]
    interpreter.set_tensor(inp["index"], x_np[None, ...])


def maybe_dequantize(output_details, y):
    scale, zp = output_details.get("quantization", (None, None))
    if scale not in (None, 0.0):
        return scale * (y.astype(np.float32) - zp)
    return y.astype(np.float32)


def run_inference(interpreter, input_size, output_details, img_path):
    x = load_and_preprocess_image(img_path, input_size)
    prepare_input(interpreter, x)

    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()

    y = interpreter.get_tensor(output_details["index"])[0]
    y = maybe_dequantize(output_details, y)

    cls = int(np.argmax(y))
    score = float(y[cls])
    infer_ms = (t1 - t0) * 1000.0
    return infer_ms, cls, score


# ---------- 显示 ----------
def init_display():
    import ST7789
    disp = ST7789.ST7789()
    disp.Init()
    disp.clear()
    disp.bl_DutyCycle(DEFAULT_BRIGHTNESS)
    return disp


def render_image(img_path, pred_text, extra_text):
    base = Image.open(img_path).resize((240, 240)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    draw.rectangle((0, 200, 240, 240), fill=(0, 0, 0, 255))
    draw.text((10, 202), pred_text[:22], font=font, fill=(255, 255, 255, 255))
    draw.text((10, 222), extra_text[:22], font=font, fill=(255, 255, 255, 255))

    return Image.alpha_composite(base, overlay).transpose(Image.ROTATE_270).convert("RGB")


# ---------- 主逻辑 ----------
def main(model_path, threads, interval):
    idx_to_name = load_categories(CATEGORIES_JSON)

    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        img_paths += glob.glob(os.path.join(TEST_DIR, ext))
    if not img_paths:
        raise RuntimeError(f"No images found in {TEST_DIR}/")

    interpreter = Interpreter(model_path=model_path, num_threads=threads)
    interpreter.allocate_tensors()
    input_size = interpreter.get_input_details()[0]["shape"][1]
    output_details = interpreter.get_output_details()[0]

    disp = init_display() if USE_DISPLAY else None

    start_time = time.time()
    start_dt = datetime.now()
    print(f"[START] Script started at {start_dt.isoformat(timespec='seconds')}")

    step = 0

    try:
        while True:
            loop_start = time.time()
            img = random.choice(img_paths)

            infer_ms, cls, score = run_inference(
                interpreter, input_size, output_details, img
            )

            elapsed_h = (time.time() - start_time) / 3600.0
            pred_name = idx_to_name.get(cls, f"class_{cls}")
            pred_text = f"{pred_name} ({score*100:.1f}%)"

            if disp:
                disp.ShowImage(
                    render_image(
                        img,
                        pred_text,
                        f"{elapsed_h:.2f}h"
                    )
                )

            print(
                f"[STEP {step:05d}] "
                f"time={datetime.now().isoformat(timespec='seconds')} | "
                f"elapsed={elapsed_h:.3f}h | "
                f"infer={infer_ms:.1f}ms"
            )

            step += 1
            time.sleep(max(0.0, interval - (time.time() - loop_start)))

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")

    finally:
        if disp:
            disp.clear()
            disp.module_exit()

        total_h = (time.time() - start_time) / 3600.0
        print(f"[END] Total runtime: {total_h:.3f}h | steps={step}")


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Battery life test via elapsed wall-clock time"
    )
    parser.add_argument("-m", "--model", default=MODEL_PATH_DEFAULT)
    parser.add_argument("-t", "--threads", type=int, default=1)
    parser.add_argument("--interval", type=float, default=INTERVAL_SEC)
    args = parser.parse_args()

    main(args.model, args.threads, args.interval)
