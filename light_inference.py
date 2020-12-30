import pickle
import cv2
from tqdm import tqdm
from time import time
from numpy import array as nparray
from numpy import asarray as npasarray
from numpy import transpose as nptranspose
from numpy import newaxis as npnewaxis

from light_classifier import Light_Classifier
from light_classifier import binary_acc
from light_classifier import Light_Dataset

import torch
import torchvision
import torch.optim as optim
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader

print(f"torch version: {torch.__version__}")
print(f"Torch CUDA version: {torch.version.cuda}")
print(f"torchvision version: {torchvision.__version__}")
print(f"opencv version: {cv2.__version__}")
print("")

#path to model to be loaded
# model_path = "E:\\Documento\\output_nn\\model#7e19.th"
# model_path = "light_classifier_v1.th"
model_path = ""

IMG_SIZE = 32

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Light_inference running on: %s"%(torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print('Light_inference running on: CPU')

# model = Light_Classifier(num_convs_backbone=convs_backbone, num_backbone_out_channels=out_channels_backbone)
model = Light_Classifier()

def load_light_weights(load_path):
    model_path = load_path
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

def light_run(img, bbox):
    bbox = list(map(int, bbox))
    img = cv2.resize(img, dsize=(512, 512))
    img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE))
    # cv2.imshow("guru", img)
    # if cv2.waitKey(1) == ord('q'):  # q to quit
    #     raise StopIteration
    img = img[npnewaxis, ...]
    img = nptranspose(img, (0, 3, 1, 2))
    img = torch.from_numpy(img).float().to(device)
    preds = model(img) 
    return preds

