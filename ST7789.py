# -*- coding:utf-8 -*-
import cv2
import time
import logging
import numpy as np
from PIL import Image

import ST7789  # 屏幕驱动

logging.basicConfig(level=logging.INFO)

# 初始化 ST7789 屏幕
disp = ST7789.ST7789()
disp.Init()
disp.clear()
disp.bl_DutyCycle(50)  # 设置背光亮度为 50%

logging.info("ST7789 初始化完成")

# 打开树莓派摄像头（官方摄像头3）
camera = cv2.VideoCapture(0)  # 0 表示第一个摄像头设备
if not camera.isOpened():
    raise RuntimeError("无法打开树莓派摄像头")

# 设置摄像头分辨率为 240x240
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

logging.info("摄像头初始化完成")

# 状态标志：True = 预览模式，False = 显示拍摄照片
preview_mode = True
captured_image = None

try:
    while True:
        if preview_mode:
            # 读取摄像头帧
            ret, frame = camera.read()
            if not ret:
                logging.warning("未获取到摄像头画面")
                continue

            # OpenCV 默认是 BGR，需要转换成 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 转换为 PIL Image 并旋转
            img_pil = Image.fromarray(frame_rgb)
            img_pil = img_pil.rotate(270)  # 旋转 270 度适配屏幕

            # 显示实时画面
            disp.ShowImage(img_pil)

        else:
            # 显示拍摄的静止照片
            if captured_image:
                disp.ShowImage(captured_image)

        # 检测 Center 按键
        center_pressed = (disp.digital_read(disp.GPIO_KEY_PRESS_PIN) == 1)
        if center_pressed:
            logging.info("Center 按键按下")
            time.sleep(0.2)  # 防抖延时

            if preview_mode:
                # 拍照：将当前帧保存为 captured_image
                captured_image = img_pil.copy()
                preview_mode = False
                logging.info("已拍照，进入照片显示模式")
            else:
                # 返回实时预览模式
                preview_mode = True
                logging.info("返回实时预览模式")

except KeyboardInterrupt:
    logging.info("用户手动退出程序")

finally:
    camera.release()
    disp.clear()
    disp.module_exit()
    logging.info("程序已退出，资源释放完成")
