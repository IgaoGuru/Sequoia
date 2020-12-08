import argparse
import os
import shutil
import time
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
# from utils.torch_utils import select_device, load_classifier, time_synchronized

# Initialize
device = torch.device("cuda:0")
print("detecting on: %s"%(torch.cuda.get_device_name(device)))

# Load model
weights = "e:\\ai\\cloud_outputs\\exp6\\weights\\best.pt"
model = attempt_load(weights, map_location=device)  # load FP32 model

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


def detect(img, weights, save_path, img_name, conf_threshold=0.4, view_img=False, save_img=False):
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
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
