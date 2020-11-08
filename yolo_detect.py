import cv2
from time import sleep
from time import time
import numpy as np
from mss import mss
from cv2 import cv2
from PIL import Image
from yolo_source import detect

weights = "e:\\ai\\cloud_outputs\\exp6\\weights\\best.pt"
save_path = "e:\\ai\\Sequoia\\runs_igor\\"

while True:
    img_name = str(int(time()*1000))
    print(img_name)
    with mss() as sct:
        # 1280 windowed mode for CS:GO, at the top left position of your main screen.
        # 26 px accounts for title bar. 
        monitor = {"top": 26, "left": 0, "width": 1280, "height": 720}
        img = sct.grab(monitor)
        #create PIL image
        img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
        img = img.resize((512, 512))
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    detect(img, weights, save_path, img_name, view_img=True) 
    # sleep(0.1)
