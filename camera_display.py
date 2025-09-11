# -*- coding:utf-8 -*-
import cv2
import time
import logging
import numpy as np
import ST7789
from PIL import Image
import RPi.GPIO as GPIO

# ========== 1. 初始化日志 ==========
logging.basicConfig(level=logging.INFO)

# ========== 2. 初始化屏幕 ==========
disp = ST7789.ST7789()       # 创建 ST7789 屏幕对象
disp.Init()                  # 初始化显示屏
disp.clear()                 # 清屏
disp.bl_DutyCycle(80)        # 设置背光亮度 50%

# ========== 3. 初始化摄像头 ==========
camera = cv2.VideoCapture(0)  # 打开摄像头
if not camera.isOpened():
    logging.error("无法打开摄像头")
    exit()

# 摄像头分辨率
CAM_WIDTH, CAM_HEIGHT = 240, 240
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

logging.info("摄像头初始化完成，分辨率: %dx%d", CAM_WIDTH, CAM_HEIGHT)

# ========== 4. 按键 GPIO 设置 ==========
# Center 按键检测
CENTER_KEY = disp.GPIO_KEY_PRESS_PIN

GPIO.setmode(GPIO.BCM)
GPIO.setup(CENTER_KEY, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ========== 5. 模式控制 ==========
MODE_PREVIEW = 0   # 实时预览模式
MODE_CAPTURED = 1  # 拍照显示模式
mode = MODE_PREVIEW

captured_image = None  # 存储拍照结果

# ========== 6. 主循环 ==========
logging.info("进入主循环，按 Center 键拍照/返回")

try:
    while True:
        # ---- 实时预览模式 ----
        if mode == MODE_PREVIEW:
            ret, frame = camera.read()
            if not ret:
                logging.error("无法读取摄像头画面")
                continue

            # 将 BGR 转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 转换为 PIL 图像
            image_pil = Image.fromarray(frame_rgb)

            # 旋转，使画面和屏幕方向一致
            image_pil = image_pil.rotate(270)

            # 显示到屏幕
            disp.ShowImage(image_pil)

            # 检测 Center 键
            if GPIO.input(CENTER_KEY) == GPIO.LOW:
                logging.info("检测到 Center 按键，拍照中...")
                captured_image = image_pil.copy()
                mode = MODE_CAPTURED
                time.sleep(0.3)  # 按键防抖

        # ---- 拍照结果显示模式 ----
        elif mode == MODE_CAPTURED:
            if captured_image:
                disp.ShowImage(captured_image)

            # 检测 Center 键返回预览
            if GPIO.input(CENTER_KEY) == GPIO.LOW:
                logging.info("返回实时预览模式")
                mode = MODE_PREVIEW
                time.sleep(0.3)  # 按键防抖

except KeyboardInterrupt:
    logging.info("退出程序")
finally:
    camera.release()
    disp.clear()
    disp.module_exit()
    GPIO.cleanup()
    logging.info("资源清理完成，程序结束")
