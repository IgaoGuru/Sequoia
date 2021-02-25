import argparse
import os
import shutil
from time import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from os import path
from numpy import random
from numpy import swapaxes
from numpy import reshape
from numpy import asarray

from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Yolo v5 running on: %s"%(torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print('Yolo v5 running on: CPU')

#init variables 
model = "to be loaded"
names, colors = None, None

# Load model
def load_yolo_weights(load_path):
    model = attempt_load(load_path, map_location=device)  # load FP32 model
    model.to(device)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

def detect(model, img, save_path, img_name, conf_threshold=0.4, view_img=False, \
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
    print(model)
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