# -*- coding:utf-8 -*-
import time
import logging
import numpy as np
from PIL import Image
import ST7789  # 屏幕驱动
from picamera2 import Picamera2  # 官方新标准库

logging.basicConfig(level=logging.INFO)

# ---------- 初始化 ST7789 ----------
disp = ST7789.ST7789()
disp.Init()
disp.clear()
disp.bl_DutyCycle(80)  # 背光亮度 80%
logging.info("ST7789 初始化完成")

# ---------- 初始化 Picamera2 ----------
picam2 = Picamera2()

# 配置摄像头输出 240x240 分辨率
config = picam2.create_preview_configuration(
    main={"size": (240, 240), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
logging.info("Picamera2 摄像头初始化完成")

# ---------- 模式状态 ----------
preview_mode = True
captured_image = None

try:
    while True:
        if preview_mode:
            # 直接获取 numpy 数组图像（RGB888 格式）
            frame_rgb = picam2.capture_array()

            # 转换为 PIL 图像并旋转
            img_pil = Image.fromarray(frame_rgb)
            img_pil = img_pil.rotate(270)

            # 显示实时画面到 ST7789
            disp.ShowImage(img_pil)

        else:
            if captured_image:
                disp.ShowImage(captured_image)

        # 检测 Center 按键
        center_pressed = (disp.digital_read(disp.GPIO_KEY_PRESS_PIN) == 1)
        if center_pressed:
            logging.info("Center 按键按下")
            time.sleep(0.2)  # 按键防抖

            if preview_mode:
                captured_image = img_pil.copy()
                preview_mode = False
                logging.info("已拍照，进入照片显示模式")
            else:
                preview_mode = True
                logging.info("返回实时预览模式")

except KeyboardInterrupt:
    logging.info("用户手动退出程序")

finally:
    picam2.stop()
    disp.clear()
    disp.module_exit()
    logging.info("程序已退出，资源释放完成")
