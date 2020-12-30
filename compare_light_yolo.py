import cv2
from tqdm import tqdm
import torch
import shutil
import argparse
import numpy as np
from ahk import AHK
from os import path
from mss import mss
from PIL import Image
from pathlib import Path
from time import sleep, time
import torch.backends.cudnn as cudnn
from random import sample
from numpy import asarray, random, reshape, swapaxes

from light_inference import light_run
from light_classifier import Light_Dataset
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
load_light_weights("light_classifier_v1.th")
print(f"using model from {weights}")

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

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
                if conf >= conf_threshold:  # Add bbox to image

                    label = '%s %.2f' % (names[int(cls)], conf)
                    bbox = [coord.item() for coord in xyxy]

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
                    else:
                        if label[:2] == "ct":
                            bboxes.append((bbox, conf.item()))
                        else:
                            bboxes.append((bbox, 1-conf.item()))

                if compare_light:
                    break #this break ensures only one bbox will be showed per viewport render (inference)

        # Stream results
        if view_img:
            cv2.imshow(p, im0)
            # if cv2.waitKey(1) == ord('q'):  # q to quit
            #     raise StopIteration
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        #plz dont use this option, thanks
        if save_img:
            cv2.imwrite(path.join(save_path, img_name) + ".png", im0)

        return bboxes


dataset = Light_Dataset("E:\\Documento\\outputs\\", 32, dlength=60000)

yolo_dict = {
    "acc" : 0,
    "precision" : 0,
    "recall" : 0,
    "f1": 0
}

light_dict = {
    "acc" : 0,
    "precision" : 0,
    "recall" : 0,
    "f1": 0
}
    
images_counted = 0
yy = 0
ll = 0

for i in tqdm(sample(range(dataset.length), dataset.length)):
    # get images/labels
    img, bboxes, labels = dataset.get_original(i)

    if len(bboxes) != 1:
        continue

    crop_img, _ = dataset[i]
    crop_img = asarray(crop_img)
    img = img.resize((512, 512))
    img = asarray(img)
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = labels[0].item()

    bboxes = detect(img, "000", "000", view_img=False, use_light=False) 

    #make sure yolo has a prediction
    if len(bboxes) !=1:
        continue
    images_counted += 1
    yolo_pred = bboxes[0][1]
    bbox = bboxes[0][0]

    #pass it onto light
    light_pred = light_run(img, bbox).item()

    if (label == 1 and yolo_pred > 0.5) or (label == 0 and yolo_pred < 0.5):
        yolo_dict["acc"] += 1
    if (label == 1 and light_pred > 0.5) or (label == 0 and light_pred < 0.5):
        light_dict["acc"] += 1
    
    if label == 1:
        if yolo_pred > light_pred:
            yy += 1
        else: ll += 1
    else:
        if yolo_pred < light_pred:
            yy += 1
        else: ll += 1

    # print("yolo: ", yolo_pred)
    # print("light: ", light_pred)
    # print(labels)
    # print("\n")

    # cv2.imshow("guru", crop_img)
    # cv2.imshow("guru2", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

print("yolo accuracy: ", yolo_dict["acc"]/images_counted)
print("light accuracy: ", light_dict["acc"]/images_counted)
print(yolo_dict["acc"], light_dict["acc"])
print(yy, ll)
print(images_counted)
