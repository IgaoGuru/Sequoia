import cv2
from time import sleep
from time import time
import numpy as np
from mss import mss
from cv2 import cv2
from PIL import Image
from yolo_source import detect

# from pynput import mouse

from ahk import AHK
ahk = AHK()

# import mouse
# import pyautogui

weights = "e:\\ai\\cloud_outputs\\exp14\\weights\\best.pt"
save_path = "e:\\ai\\Sequoia\\runs_igor\\"

window_x = 1280
window_y = 720
y_offset = 26
window_shape = [window_x, window_y, y_offset]

def shoot(bbox):
    #AKH :: 78 = 90degrees
        #   39 = 30 (ao inves de 26)

    bbox = list(map(int, bbox))
    bbox[0] = bbox[0]*1280/512 
    bbox[2] = bbox[2]*1280/512 
    
    bbox[1] = bbox[1]*720/512 
    bbox[3] = bbox[3]*720/512 

    x = (((bbox[2]-bbox[0])/2) + bbox[0]) - 640
    x_m = -0.00005*(x**2)  + 0.1094 * x

    #   target: body
    # y = (((bbox[3]-bbox[1])/2) + bbox[1]) - 360
    #   target: head
    y = (bbox[1] + 20) - 360
    y_m = -0.00005*(y**2)  + 0.0463 * y

    if x<= 20 and  y <= 20: 
        ahk.click()

    print("\n")
    print(bbox)
    print(x)
    print(x_m)
    print(y)
    print(y_m)

    if x_m != 0 and y_m != 0:
        ahk.mouse_move(x=x_m, y=y_m, blocking=True,\
            speed=0, relative=False)


while True:
    img_name = str(int(time()*1000))
    with mss() as sct:
        # 1280 windowed mode for CS:GO, at the top left position of your main screen.
        # 26 px accounts for title bar. 
        monitor = {"top": y_offset, "left": 0, "width": window_x, "height": window_y}
        img = sct.grab(monitor)
        #create PIL image
        img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
        img = img.resize((512, 512))
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes = detect(img, weights, save_path, img_name, view_img=True, save_img=False) 

        if len(bboxes) > 0:
            shoot(bboxes[0])
            print(bboxes)

        # sleep(0.01)

# shoot_2([1,2,3,4])