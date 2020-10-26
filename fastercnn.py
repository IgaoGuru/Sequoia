from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def get_fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True, num_classes=2):
    torchvision.models.detection.fasterrcnn_resnet50_fpn()
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        pretrained_backbone=pretrained_backbone,
        num_classes=num_classes)


def get_custom_fasterrcnn(num_classes=2):
    backbone = [
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(num_features=32),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ]
    backbone = nn.Sequential(*backbone)
    backbone.out_channels = 32

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=5,
                                                    sampling_ratio=2)

    return FasterRCNN(backbone,
                      num_classes=num_classes,
                      rpn_anchor_generator=anchor_generator,
                      box_roi_pool=roi_pooler)

def get_simple_backbone(num_convs=3, num_out_channels=32):
    kernel_size = 3
    padding = (kernel_size - 1) // 2
    backbone_layers = [
        nn.Conv2d(3, num_out_channels, kernel_size=kernel_size, stride=2, padding=padding, groups=1, bias=False),
        nn.BatchNorm2d(num_out_channels),
        nn.ReLU(inplace=True)
    ]
    for i in range(num_convs - 1):
        backbone_layers.append(nn.Conv2d(num_out_channels, num_out_channels, kernel_size=kernel_size, stride=2, padding=padding, groups=1, bias=False))
        backbone_layers.append(nn.BatchNorm2d(num_out_channels))
        backbone_layers.append(nn.ReLU(inplace=True))
    backbone = nn.Sequential(*backbone_layers)
    backbone.out_channels = num_out_channels 
    return backbone

def get_fasterrcnn_small(num_classes=2, num_convs_backbone=3, num_backbone_out_channels=32):
    # backbone = get_simple_backbone(num_convs=num_convs_backbone, num_out_channels=num_backbone_out_channels)
    backbone = torchvision.models.alexnet(pretrained=True).features
    backbone.out_channels = 256
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),), #BUG: doesn't have backbone
                                       aspect_ratios=((0.25, 0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=5,
                                                    sampling_ratio=2)
    return FasterRCNN(backbone,
                      num_classes=num_classes,
                      rpn_anchor_generator=anchor_generator,
                      box_roi_pool=roi_pooler)

# put the pieces together inside a FasterRCNN model
def get_fasterrcnn_mobile(pretrained_backbone=True, num_classes=2):
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=pretrained_backbone).features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),
                                       aspect_ratios=((0.25, 0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=5,
                                                    sampling_ratio=2)

    return FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

CSGO_CATEGORY_NAMES = ['__background__', 'Terrorist', 'CounterTerrorist']

COCO_INSTANCE_CATEGORY_COLORS = {}
for i, v in enumerate(COCO_INSTANCE_CATEGORY_NAMES):
    COCO_INSTANCE_CATEGORY_COLORS[v] = list(np.random.randint(0, 255, (3, )))

def get_coco_category_color_mask(image, coco_category):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = COCO_INSTANCE_CATEGORY_COLORS[coco_category]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_prediction_fastercnn(img_path, model, threshold, category_names=None, img_is_path=True):
    if category_names is None:
        category_names = COCO_INSTANCE_CATEGORY_COLORS
    if img_is_path:
        img = Image.open(img_path).convert("RGB")
    else:
        img = img_path
    pred = model([img])
    #print(pred)
    pred_boxes = list(pred[0]['boxes'].detach().cpu().numpy())
    pred_labels = list(pred[0]['labels'].detach().cpu().numpy())
    pred_scores = list(pred[0]['scores'].detach().cpu().numpy())
    #print(f"Pred. boxes: {pred_boxes}")
    #print(f"Pred. labels: {pred_labels}")
    #print(f"Pred. scores: {pred_scores}")

    pred_t_list = [pred_scores.index(x) for x in pred_scores if x > threshold]
    if len(pred_t_list) == 0:
        return None, None, None
    pred_t = pred_t_list[-1]
    pred_class = [category_names[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class, pred_scores[:pred_t+1]

def detect_on_img(img, net, threshold, text_size=1, text_th=1, rect_th=1):
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masks, boxes, pred_cls, pred_scores = get_prediction_fastercnn(img, net, threshold, img_is_path=False)
    if masks is None:
        return img
    for i in range(len(masks)):
        # rgb_mask = random_colour_masks(masks[i])
        if len(masks[i].shape) < 2:
            continue
        rgb_mask = get_coco_category_color_mask(masks[i], pred_cls[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        img = cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        #bbox_text = f"{pred_cls[i]}: {pred_scores[i]:.2f} >= {threshold:.2f}"
        bbox_text = f"{pred_cls[i]}: {pred_scores[i]:.2f}"
        img = cv2.putText(img, bbox_text,
                          boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,
                          text_size, (0, 255, 0), thickness=text_th)
    return img