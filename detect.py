import cv2
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
from light_inference import load_light_weights
from models.experimental import attempt_load
from utils.general import (apply_classifier, check_img_size,
                           non_max_suppression, plot_one_box, scale_coords,
                           set_logging, strip_optimizer, xyxy2xywh)


ahk = AHK()

parser = argparse.ArgumentParser(description='Detect on CS:GO')
parser.add_argument('-w', help='absolute path to custom weights for YOLO(optional)', type=str, nargs='?', default='sequoiaV1.pt')
parser.add_argument('-wl', help='absolute path to custom weights for Light_Classifier', type=str, nargs='?', default='light_classifierV1.th')
parser.add_argument('-s', help='absolute path to directory where images from detection can be saved (optional)', type=str, nargs='?', default=None)
parser.add_argument('-x', help='the x component of your game\'s resolution eg.([1280] x 720)', type=int, nargs='?', default=1280)
parser.add_argument('-y', help='the x component of your game\'s resolution eg.(1280 x [720])', type=int, nargs='?', default=720)
parser.add_argument('-off', help='the height of your game\'s window bar at the top (to be compensated)', type=int, nargs='?', default=26)
parser.add_argument('-shoot', help='toggles auto-shooting (i.e. automatic mouse movement) [either 0 (off) or 1 (on)]', type=int, nargs='?', default=False)
parser.add_argument('-bench', help='toggles benchmark mode (displays inference times in ms) [either 0 (off) or 1 (on)]', type=int, nargs='?', default=False)
args = parser.parse_args()

weights = args.w
save_path = args.s
window_x = args.x
window_y = args.y
y_offset = args.off
_shoot = args.shoot
benchmark = args.bench
print(_shoot, benchmark)

window_shape = [window_x, window_y, y_offset]

weights = "e:\\ai\\cloud_outputs\\exp14\\weights\\best.pt"

# Initialize
device = torch.device("cuda:0")
print("detecting on: %s"%(torch.cuda.get_device_name(device)))

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
load_light_weights(args.wl)
print(f"using model from {weights}")

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


def shoot(bbox):
    """Manages bbox to mouse emulator input conversion and clicking.

    Args:
        bbox (list): list of four integers representing the pixel coordinates of the enemy [x0,y0,x1,y1].
    """

    ## note to myself: auto-shooting still very experimental
    #AKH :: 78 = 90degrees
        #   39 = 30 (ao inves de 26)

    bbox = list(map(int, bbox))
    bbox[0] = bbox[0]*window_x/512 
    bbox[2] = bbox[2]*window_x/512 
    
    bbox[1] = bbox[1]*window_y/512 
    bbox[3] = bbox[3]*window_y/512 

    x = (((bbox[2]-bbox[0])/2) + bbox[0]) - int(window_x/2)
    x_m = -0.00005*(x**2)  + 0.1094 * x

    #   target: body
    # y = (((bbox[3]-bbox[1])/2) + bbox[1]) - 360
    #   target: head
    y = (bbox[1] + 20) - 360
    y_m = -0.00005*(y**2)  + 0.0463 * y

    if x<= 20 and  y <= 20: 
        ahk.click()

    if x_m != 0 and y_m != 0:
        ahk.mouse_move(x=x_m, y=y_m, blocking=True,\
            speed=0, relative=False)

def detect(img, save_path, img_name, conf_threshold=0.4, view_img=False, \
    save_img=False, use_light=True, compare_light=False, only_enemies=False, enemy_str=None, benchmark=False):
    """Manage all aspects of inference for both networks (yolo and light).

    Args:
        img (numpy.ndarray): image to be processed.
        save_path (str): percentage of dataset to be allocated to validation.
        img_name (str): base name for images to be saved [if save_img is enabled].
        conf_threshold (float): threshold to which bounding boxes will not be considered.
        view_img (bool): option for visualizing the outputs of the NN in real time.
        save_img (bool): option for saving images images with labeling.
        use_light (bool): toggles the use of a helper NN, light_classifier, that aids in the classification of enemies/allies. 
        compare_light (bool): toggles real time viewport annotation of comparison between yolo's classification and light's classification.
        only_enemies (bool): toggles only considering enemy bounding boxes.
        enemy_str (str): string that corresponds to which team is the enemy to be considered [in case only_enemies is enabled]; can be either "ct" or "tr".
        benchmark (bool): toggles benchmarking mode, in which inference times will be printed on the console.
    Returns:
        list of found bounding boxes.
    """
    if only_enemies and enemy_str==None:
        raise Exception("You should declare which string represents your enemy! (either \"ct\" or \"tr\")")

    # Run inference
    im0 = img #save raw image for later
    ## preparing img for torch inference
    img = swapaxes(img, 0, 2)
    img = swapaxes(img, 1, 2)
    img = img.reshape(1, 3, 512, 512)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # -- Inference --
    tic_yolo = time()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred) 
    toc_yolo = (time() - tic_yolo)*1000
    if benchmark:
        print(f"yolo: {toc_yolo} ms")

    # Process detections
    bboxes = []
    for i, det in enumerate(pred):  # detections per image
        p, s = "teste", ""

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
                    bbox = [coord.item() for coord in xyxy]
                    bboxes.append(bbox)

                    ## get predictions for light_classifier
                    tic_light = time()
                    light_pred = light_run(im0, bbox).item()
                    toc_light = (time() - tic_light)*1000
                    if benchmark:
                        print(f"light: {toc_light} ms")

                    # plot the bboxes on image
                    if use_light:
                        if light_pred >= 0.5: ct_tr_light = "ct"
                        else: ct_tr_light = "tr"

                        label_light = f"{ct_tr_light}, {light_pred:3f}"
                        plot_one_box(xyxy, im0, label=label_light, color=colors[int(cls)], line_thickness=2)
                    else:
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                    if only_enemies:
                        if (use_light and ct_tr_light==enemy_str) or \
                        (not use_light and label[:2]==enemy_str):
                            bboxes.append(bbox)
                    else: bboxes.append(bbox)

                    if compare_light:
                        if label[:2] != "ct" and label[:2] != "tr":
                            cv2.putText(im0,"nothing", (10,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        else:
                            if light_pred >= 0.5:
                                ct_tr_light = "ct"
                            else: ct_tr_light = "tr"
                            cv2.putText(im0,f"yolo:{label[:2]}, light:{light_pred:1f} ({ct_tr_light})", (10,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                if compare_light:
                    break #this break ensures only one bbox will be showed per viewport render (inference)

        # Stream results
        if view_img:
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
        #plz dont use this option, thanks
        if save_img:
            cv2.imwrite(path.join(save_path, img_name) + ".png", im0)

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

        tic = time()
        bboxes = detect(imgarr, save_path, img_name, view_img=True, benchmark=benchmark) 
        toc = time() - tic
        if benchmark:
            print(f'total time: {toc*1000:1f} ms')

        if len(bboxes) > 0:
            if _shoot:
                shoot(bboxes[0])

        # sleep(0.01)

            