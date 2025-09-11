# -*- coding:utf-8 -*-
import cv2
import time
import logging
import numpy as np
import ST7789
from PIL import Image

# ========== 1. 初始化日志 ==========
logging.basicConfig(level=logging.INFO)

# ========== 2. 初始化屏幕 ==========
disp = ST7789.ST7789()       # 创建 ST7789 屏幕对象
disp.Init()                  # 初始化显示屏
disp.clear()                 # 清屏
disp.bl_DutyCycle(80)        # 设置背光亮度 80%

# ========== 3. 初始化摄像头 ==========
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    logging.error("无法打开摄像头")
    exit()

CAM_WIDTH, CAM_HEIGHT = 320, 240
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
logging.info("摄像头初始化完成，分辨率: %dx%d", CAM_WIDTH, CAM_HEIGHT)

# ========== 4. 模式控制 ==========
MODE_PREVIEW = 0   # 实时预览模式
MODE_CAPTURED = 1  # 拍照显示模式
mode = MODE_PREVIEW

captured_image = None

logging.info("进入主循环，按 Center 键拍照/返回")

try:
    while True:
        if mode == MODE_PREVIEW:
            ret, frame = camera.read()
            if not ret:
                logging.error("无法读取摄像头画面")
                continue

            # 将 BGR 转为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 缩放到 ST7789 分辨率
            frame_rgb = cv2.resize(frame_rgb, (disp.width, disp.height))

            # 转换为 PIL 图像
            image_pil = Image.fromarray(frame_rgb)
            image_pil = image_pil.rotate(270)  # 根据实际情况调整

            # 显示到屏幕
            disp.ShowImage(image_pil)

            # 检测 Center 按键
            if disp.digital_read(disp.GPIO_KEY_PRESS_PIN) == 1:  # 1 表示按下
                logging.info("检测到 Center 按键，拍照中...")
                captured_image = image_pil.copy()
                mode = MODE_CAPTURED
                time.sleep(0.3)  # 按键防抖

        elif mode == MODE_CAPTURED:
            if captured_image:
                disp.ShowImage(captured_image)

            # 再次按下 Center 键返回实时预览
            if disp.digital_read(disp.GPIO_KEY_PRESS_PIN) == 1:
                logging.info("返回实时预览模式")
                mode = MODE_PREVIEW
                time.sleep(0.3)

except KeyboardInterrupt:
    logging.info("退出程序")
finally:
    camera.release()
    disp.clear()
    disp.module_exit()
    logging.info("资源清理完成，程序结束")
