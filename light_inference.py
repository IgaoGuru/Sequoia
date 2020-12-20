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
model_path = "E:\\Documento\\output_nn\\model#7e19.th"
IMG_SIZE = 28

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on: %s"%(torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print('running on: CPU')

# model = Light_Classifier(num_convs_backbone=convs_backbone, num_backbone_out_channels=out_channels_backbone)
model = Light_Classifier()
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

def light_run(img, bbox):
    img = img.crop(tuple(bbox))
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = nparray(img) 
    img = nptranspose(img, (2, 0, 1))
    img = img[npnewaxis, ...]
    img = torch.from_numpy(img).float().to(device)

    preds = model(img) 
    print("LIGHT PREDS: ", preds)

