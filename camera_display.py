# -*- coding:utf-8 -*-
import cv2
import time
import logging
from PIL import Image
import ST7789  # 屏幕驱动

logging.basicConfig(level=logging.INFO)

# ---------- 初始化 ST7789 ----------
disp = ST7789.ST7789()
disp.Init()
disp.clear()
disp.bl_DutyCycle(80)  # 背光亮度 80%
logging.info("ST7789 初始化完成")

# ---------- rpicam 管道 ----------
pipeline = (
    "libcamerasrc ! "
    "video/x-raw,width=240,height=240,framerate=30/1 ! "
    "videoconvert ! "
    "appsink drop=true sync=false"
)

camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not camera.isOpened():
    raise RuntimeError("无法通过 rpicam 管道打开树莓派摄像头")
logging.info("rpicam 摄像头初始化完成")

# ---------- 模式状态 ----------
preview_mode = True
captured_image = None

try:
    while True:
        if preview_mode:
            ret, frame = camera.read()
            if not ret:
                logging.warning("未获取到摄像头画面")
                continue

            # OpenCV 默认 BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 转为 PIL 并旋转
            img_pil = Image.fromarray(frame_rgb)
            img_pil = img_pil.rotate(270)

            # 显示实时画面
            disp.ShowImage(img_pil)

        else:
            if captured_image:
                disp.ShowImage(captured_image)

        # 检测 Center 按键
        center_pressed = (disp.digital_read(disp.GPIO_KEY_PRESS_PIN) == 1)
        if center_pressed:
            logging.info("Center 按键按下")
            time.sleep(0.2)  # 防抖

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
    camera.release()
    disp.clear()
    disp.module_exit()
    logging.info("程序已退出，资源释放完成")
