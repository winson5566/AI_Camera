# -*- coding: utf-8 -*-
import os
import time
import json
import random
import glob
import csv
import subprocess
from datetime import datetime
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 适配树莓派：优先使用 tflite-runtime
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ----------------- 配置 -----------------
MODEL_PATH_DEFAULT = "model/model_efficientnet_b0_inat2021_drq.tflite"
CATEGORIES_JSON = "inat2021/categories.json"
TEST_DIR = "test"
CENTER_CROP = True

# 每 10 秒跑一次推理
INTERVAL_SEC = 10.0

# 显示
USE_DISPLAY = True
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 18
DEFAULT_BRIGHTNESS = 100

# 电量判停阈值（两者满足其一就停）
STOP_BATTERY_PERCENT = 1.0     # <= 1% 认为快没电
STOP_BATTERY_VOLT = 3.30       # <= 3.30V 认为快没电（按你电池/UPS实际调）

# 记录
LOG_DIR = "battery_logs"
# ---------------------------------------

# ---------- 工具函数 ----------
def load_categories(categories_json):
    with open(categories_json, 'r', encoding='utf-8') as f:
        cats = json.load(f)
    idx_to_name = {}
    for i, c in enumerate(cats):
        idx = int(c.get('id', i))
        name = c.get('common_name') or c.get('name') or f"class_{idx}"
        idx_to_name[idx] = name
    return idx_to_name

def fix_to_uint8(x_np: np.ndarray) -> np.ndarray:
    if x_np.dtype.kind == 'f':
        if float(x_np.max()) <= 1.0 and float(x_np.min()) >= 0.0:
            x_np = np.round(x_np * 255.0)
        x_np = np.clip(x_np, 0.0, 255.0).astype(np.uint8, copy=False)
    else:
        x_np = np.clip(x_np, 0, 255).astype(np.uint8, copy=False)
    return x_np

def load_and_preprocess_image(path, size, center_crop=True) -> np.ndarray:
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

def prepare_input(interpreter, x_np: np.ndarray):
    input_details = interpreter.get_input_details()[0]
    in_index = input_details['index']
    in_dtype = input_details['dtype']
    wanted_shape = list(input_details['shape'])

    x_u8 = fix_to_uint8(x_np)
    target_shape = [1, x_u8.shape[0], x_u8.shape[1], x_u8.shape[2]]

    if wanted_shape != target_shape:
        interpreter.resize_tensor_input(in_index, target_shape, strict=False)
        interpreter.allocate_tensors()

    if in_dtype == np.uint8:
        x_for_model = x_u8[None, ...]
    elif in_dtype == np.int8:
        scale, zero_point = input_details.get('quantization', (None, None))
        if scale in (None, 0.0):
            x_q = x_u8.astype(np.int32) - 128
            x_q = np.clip(x_q, -128, 127).astype(np.int8, copy=False)
        else:
            x_q = np.round(x_u8.astype(np.float32) / float(scale) + float(zero_point))
            x_q = np.clip(x_q, -128, 127).astype(np.int8, copy=False)
        x_for_model = x_q[None, ...]
    elif in_dtype == np.float32:
        x_for_model = x_u8.astype(np.float32, copy=False)[None, ...]
    else:
        raise ValueError(f"Unsupported input dtype: {in_dtype}")

    interpreter.set_tensor(in_index, x_for_model)

def maybe_dequantize_output(output_details, y):
    y = y.astype(np.float32, copy=False)
    scale, zero_point = output_details.get('quantization', (None, None))
    if scale not in (None, 0.0):
        y = scale * (y - float(zero_point))
    return y

def run_inference_on_image(interpreter, input_size, output_details, img_path):
    arr = load_and_preprocess_image(img_path, size=input_size, center_crop=CENTER_CROP)
    prepare_input(interpreter, arr)

    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()
    infer_ms = (t1 - t0) * 1000.0

    y = interpreter.get_tensor(output_details['index'])
    if y.ndim == 2 and y.shape[0] == 1:
        y = y[0]
    y = maybe_dequantize_output(output_details, y)

    cls = int(np.argmax(y))
    score = float(y[cls])
    return infer_ms, cls, score

# ---------- PiSugar 电池读取（8423 端口） ----------
def _pigsugar_cmd(cmd: str) -> Optional[str]:
    """
    PiSugar 常见：echo "get battery_v" | nc 127.0.0.1 8423
    返回形如：battery_v: 4.178834
    """
    try:
        out = subprocess.check_output(
            ["bash", "-lc", f'echo "{cmd}" | nc 127.0.0.1 8423'],
            stderr=subprocess.STDOUT,
            timeout=1.5,
        ).decode("utf-8", errors="ignore").strip()
        return out
    except Exception:
        return None

def read_battery_percent() -> Optional[float]:
    out = _pigsugar_cmd("get battery")
    if not out:
        return None
    # battery: 87
    try:
        val = out.split("battery:")[1].strip()
        return float(val)
    except Exception:
        return None

def read_battery_voltage() -> Optional[float]:
    out = _pigsugar_cmd("get battery_v")
    if not out:
        return None
    # battery_v: 4.1759667
    try:
        val = out.split("battery_v:")[1].strip()
        return float(val)
    except Exception:
        return None

# ---------- 显示 ----------
def init_display():
    import ST7789
    disp = ST7789.ST7789()
    disp.Init()
    disp.clear()
    disp.bl_DutyCycle(DEFAULT_BRIGHTNESS)
    return disp

def render_result_image(img_path: str, pred_text: str, extra_text: str = "") -> Image.Image:
    """
    参考你拍照程序：底部黑条 + 白字；并 ROTATE_270 适配 ST7789
    """
    base = Image.open(img_path).convert("RGB").resize((240, 240), Image.BILINEAR).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # 底部黑条
    draw.rectangle((0, 200, 240, 240), fill=(0, 0, 0, 255))
    draw.text((10, 202), pred_text[:22], font=font, fill=(255, 255, 255, 255))
    if extra_text:
        draw.text((10, 222), extra_text[:22], font=font, fill=(255, 255, 255, 255))

    overlay = overlay.transpose(Image.ROTATE_270)
    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out

# ---------- 主逻辑 ----------
def main(model_path: str, threads: int, interval_sec: float, auto_shutdown: bool):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"battery_life_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    idx_to_name = load_categories(CATEGORIES_JSON)
    print(f"[INFO] Loaded {len(idx_to_name)} categories.")
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Threads: {threads}")
    print(f"[INFO] Interval: {interval_sec}s")

    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(TEST_DIR, ext)))
    if not img_paths:
        raise RuntimeError(f"No images found in {TEST_DIR}/")

    # 载入模型
    t0 = time.perf_counter()
    interpreter = Interpreter(model_path=model_path, num_threads=int(threads))
    interpreter.allocate_tensors()
    t1 = time.perf_counter()
    load_ms = (t1 - t0) * 1000.0

    input_size = int(interpreter.get_input_details()[0]['shape'][1])
    output_details = interpreter.get_output_details()[0]
    print(f"[INFO] Model loaded in {load_ms:.2f} ms | input={input_size}x{input_size}")

    # 初始化屏幕
    disp = None
    if USE_DISPLAY:
        try:
            disp = init_display()
            print("[INFO] ST7789 display initialized.")
        except Exception as e:
            print(f"[WARN] Display init failed: {e}")
            disp = None

    # CSV header
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp",
            "battery_percent",
            "battery_v",
            "img",
            "infer_ms",
            "cls",
            "score",
            "threads",
        ])

    print(f"[INFO] Logging to: {log_path}")
    print("[INFO] Running battery life test... (Ctrl+C to stop)\n")

    start_time = time.time()
    step = 0

    try:
        while True:
            step_start = time.time()

            # 读电池
            bat_pct = read_battery_percent()
            bat_v = read_battery_voltage()

            # 随机选图
            img_path = random.choice(img_paths)

            # 推理
            infer_ms, cls, score = run_inference_on_image(
                interpreter, input_size, output_details, img_path
            )
            pred_name = idx_to_name.get(cls, f"Class {cls}")

            # 显示文案（你喜欢的风格：名字 + 置信度）
            pred_text = f"{pred_name} ({score * 100:.1f}%)"
            extra = []
            if bat_pct is not None:
                extra.append(f"BAT {bat_pct:.0f}%")
            if bat_v is not None:
                extra.append(f"{bat_v:.3f}V")
            extra_text = "  ".join(extra)

            # 屏幕显示
            if disp is not None:
                try:
                    img_out = render_result_image(img_path, pred_text, extra_text)
                    disp.ShowImage(img_out)
                except Exception as e:
                    print(f"[WARN] Display show failed: {e}")

            # 记录
            ts = datetime.now().isoformat(timespec="seconds")
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    ts,
                    "" if bat_pct is None else f"{bat_pct:.2f}",
                    "" if bat_v is None else f"{bat_v:.6f}",
                    os.path.basename(img_path),
                    f"{infer_ms:.2f}",
                    cls,
                    f"{score:.6f}",
                    threads,
                ])

            elapsed = time.time() - start_time
            step += 1
            print(f"[{step:05d}] {ts} | {os.path.basename(img_path)} | "
                  f"{pred_text} | {infer_ms:.2f} ms | "
                  f"BAT%={bat_pct} V={bat_v} | elapsed={elapsed/3600:.2f}h")

            # 判停：电压 or 百分比
            low_by_pct = (bat_pct is not None and bat_pct <= STOP_BATTERY_PERCENT)
            low_by_v = (bat_v is not None and bat_v <= STOP_BATTERY_VOLT)

            if low_by_pct or low_by_v:
                reason = []
                if low_by_pct: reason.append(f"battery% <= {STOP_BATTERY_PERCENT}")
                if low_by_v: reason.append(f"battery_v <= {STOP_BATTERY_VOLT}")
                print(f"\n[STOP] Low battery detected: {', '.join(reason)}")
                break

            # 对齐 10 秒周期（把推理耗时也算进去）
            spent = time.time() - step_start
            sleep_s = max(0.0, interval_sec - spent)
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")

    finally:
        if disp is not None:
            try:
                disp.clear()
                disp.module_exit()
            except Exception:
                pass

        total = time.time() - start_time
        print(f"[INFO] Done. Total runtime: {total/3600:.2f} hours | steps: {step}")
        print(f"[INFO] Log saved: {log_path}")

        if auto_shutdown:
            print("[INFO] Auto shutdown enabled: sudo shutdown now")
            os.system("sudo shutdown now")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Battery life test: run inference every N seconds until battery dies.")
    parser.add_argument("-m", "--model", type=str, default=MODEL_PATH_DEFAULT, help="TFLite model path")
    parser.add_argument("-t", "--threads", type=int, default=1, help="TFLite num_threads")
    parser.add_argument("--interval", type=float, default=INTERVAL_SEC, help="Seconds per inference (default=10)")
    parser.add_argument("--shutdown", action="store_true", help="Shutdown when low battery detected")
    args = parser.parse_args()

    main(args.model, args.threads, args.interval, args.shutdown)
