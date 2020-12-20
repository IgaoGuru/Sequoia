import cv2
import time
import torch
import shutil
import argparse
import numpy as np
from ahk import AHK
from os import path
from mss import mss
from cv2 import cv2
from PIL import Image
from pathlib import Path
from time import sleep, time
import torch.backends.cudnn as cudnn
from numpy import asarray, random, reshape, swapaxes

from light_inference import light_run
from models.experimental import attempt_load
from utils.general import (apply_classifier, check_img_size,
                           non_max_suppression, plot_one_box, scale_coords,
                           set_logging, strip_optimizer, xyxy2xywh)


ahk = AHK()

parser = argparse.ArgumentParser(description='Detect on CS:GO')
parser.add_argument('-w', help='absolute path to location of custom weights (optional)', type=str, nargs='?', default='sequoiaV1.pt')
parser.add_argument('-s', help='absolute path to directory where images from detection can be saved (optional)', type=str, nargs='?', default=None)
parser.add_argument('-x', help='the x component of your game\'s resolution eg.([1280] x 720)', type=int)
parser.add_argument('-y', help='the x component of your game\'s resolution eg.(1280 x [720])', type=int)
parser.add_argument('-off', help='the height of your game\'s window bar at the top (to be compensated)', type=int, nargs='?', default=26)
args = parser.parse_args()

weights = args.w
save_path = args.s
window_x = args.x
window_y = args.y
y_offset = args.off

window_shape = [window_x, window_y, y_offset]
# weights = "e:\\ai\\cloud_outputs\\exp14\\weights\\best.pt"
# save_path = "e:\\ai\\Sequoia\\runs_igor\\"

# Initialize
device = torch.device("cuda:0")
print("detecting on: %s"%(torch.cuda.get_device_name(device)))

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
print(f"using model from {weights}")

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


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

def detect(img, save_path, img_name, conf_threshold=0.4, view_img=False, save_img=False):
    # out, source, weights, view_img, save_txt, imgsz = \
    #     opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Run inference
    # print("2::", img.shape)
    im0 = img #save raw image for later
    img = swapaxes(img, 0, 2)
    img = swapaxes(img, 1, 2)
    img = img.reshape(1, 3, 512, 512)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # Inference
    # pred = model(img, augment=opt.augment)[0]
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred) 

    # Process detections
    bboxes = []
    for i, det in enumerate(pred):  # detections per image
        p, s = "teste", ''

        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if (save_img or view_img) and conf >= conf_threshold:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                    bbox = []
                    #only return enemy bboxes:
                        # if label[:2] == "ct":
                        #     for coord in xyxy:
                        #         bbox.append(coord.item())
                        #     bboxes.append(bbox)
                    for coord in xyxy:
                        bbox.append(coord.item())
                    bboxes.append(bbox)

        # Print time (inference + NMS)
        # print('%sDone.' % (s))

        # Stream results
        if view_img:
            # im0 = cv2.resize(im0, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
        if save_img:
            cv2.imwrite(path.join(save_path, img_name) + ".png", im0)

        if save_img or view_img:
            return bboxes

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
        imgarr = np.asarray(img)
        imgarr = cv2.cvtColor(imgarr, cv2.COLOR_BGR2RGB)

        bboxes = detect(imgarr, save_path, img_name, view_img=True, save_img=False) 

        if len(bboxes) > 0:
            shoot(bboxes[0])
            light_run(img, bboxes[0])
            # print(bboxes)

        # sleep(0.01)

            