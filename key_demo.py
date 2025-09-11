# 1. 引入库和初始化显示屏

# -*- coding:utf-8 -*-
import spidev as SPI
import logging
import ST7789

import time
import subprocess

from PIL import Image,ImageDraw,ImageFont

# 240x240 display with hardware SPI:
disp = ST7789.ST7789() # 创建 ST7789 屏幕对象
disp.Init() # 初始化显示屏

# Clear display.
disp.clear() # 清屏

#Set the backlight to 100
disp.bl_DutyCycle(50) # 设置背光亮度为 50%

# 2. 创建一个空白画布用于绘图
# Create blank image for drawing.
# Make sure to create image with mode '1' for 1-bit color.
image1 = Image.new("RGB", (disp.width, disp.height), "WHITE")

# Get drawing object to draw on image.
draw = ImageDraw.Draw(image1)

# Draw a black filled box to clear the image.
draw.rectangle((0,0,disp.width, disp.height), outline=0, fill=0)
disp.ShowImage(image1)

# 🔁 3. 主循环：不断检测每个按键的状态
try:
    while True:
        # with canvas(device) as draw:
        if disp.digital_read(disp.GPIO_KEY_UP_PIN ) == 0: # button is released
            draw.polygon([(20, 20), (30, 2), (40, 20)], outline=255, fill=0xff00)  #Up        
        else: # button is pressed:
            draw.polygon([(20, 20), (30, 2), (40, 20)], outline=255, fill=0)  #Up filled
            print ("Up")
            
        if disp.digital_read(disp.GPIO_KEY_LEFT_PIN) == 0: # button is released
            draw.polygon([(0, 30), (18, 21), (18, 41)], outline=255, fill=0xff00)  #left           
        else: # button is pressed:
            draw.polygon([(0, 30), (18, 21), (18, 41)], outline=255, fill=0)  #left filled
            print ("left") 
            
        if disp.digital_read(disp.GPIO_KEY_RIGHT_PIN) == 0: # button is released
            draw.polygon([(60, 30), (42, 21), (42, 41)], outline=255, fill=0xff00) #right        
        else: # button is pressed:
            draw.polygon([(60, 30), (42, 21), (42, 41)], outline=255, fill=0) #right filled
            print ("right")
            
        if disp.digital_read(disp.GPIO_KEY_DOWN_PIN) == 0: # button is released
            draw.polygon([(30, 60), (40, 42), (20, 42)], outline=255, fill=0xff00) #down        
        else: # button is pressed:
            draw.polygon([(30, 60), (40, 42), (20, 42)], outline=255, fill=0) #down filled
            print ("down")
            
        if disp.digital_read(disp.GPIO_KEY_PRESS_PIN) == 0: # button is released
            draw.rectangle((20, 22,40,40), outline=255, fill=0xff00) #center         
        else: # button is pressed:
            draw.rectangle((20, 22,40,40), outline=255, fill=0) #center filled
            print ("center")
            
        if disp.digital_read(disp.GPIO_KEY1_PIN) == 0: # button is released
            draw.ellipse((70,0,90,20), outline=255, fill=0xff00) #A button        
        else: # button is pressed:
            draw.ellipse((70,0,90,20), outline=255, fill=0) #A button filled
            print ("KEY1")
            
        if disp.digital_read(disp.GPIO_KEY2_PIN) == 0: # button is released
            draw.ellipse((100,20,120,40), outline=255, fill=0xff00) #B button]        
        else: # button is pressed:
            draw.ellipse((100,20,120,40), outline=255, fill=0) #B button filled
            print ("KEY2")
            
        if disp.digital_read(disp.GPIO_KEY3_PIN) == 0: # button is released
            draw.ellipse((70,40,90,60), outline=255, fill=0xff00) #A button        
        else: # button is pressed:
            draw.ellipse((70,40,90,60), outline=255, fill=0) #A button filled
            print ("KEY3")
        disp.ShowImage(image1)
except:
	print("except")
disp.module_exit()


