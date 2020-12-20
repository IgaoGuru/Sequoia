'''
use this file to find the mean and std from any Csgo-Data dataset
'''

import pickle
import cv2
from time import time
from tqdm import tqdm
from time import time
from numpy import array as nparray
from numpy import asarray as npasarray
from numpy import transpose as nptranspose
from numpy import mean as npmean
from numpy import std as npstd
import numpy as np

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

SEED = 21
IMG_SIZE_X = 1280
IMG_SIZE_Y = 720

torch.manual_seed(SEED)
n_img_size = 100
num_epochs = 500
scale_factor = 1
# checkpoints = [14, 19, 49, 79, 99, 119, 149, 179, 199] #all epoch indexes where the network should be saved
checkpoints = [0, 14, 19, 49, 79, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
model_number = 10 #currently using '999' as "disposable" model_number :)
batch_size = 1
# batch_size = 2
convs_backbone = 1
out_channels_backbone = 4
reg_weight = 1 # leave 1 for no weighting
dlength = 1000 # leave None for maximum dataset length

# dataset_path = "C:\\Users\\User\\Documents\\GitHub\\Csgo-NeuralNetworkPaulo\\data\\datasets\\"  #remember to put "/" at the end
dataset_path = "E:\\Documento\\outputs\\"  #remember to put "\\" at the end
# dataset_path = "/home/sequoia/data/"
model_save_path = 'E:\\Documento\\output_nn\\'

#OPTIMIZER PARAMETERS ###############
lr = 0.1 
lrs = [1e-2, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5]
lr_idx = 0
weight_decay = 0

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on: %s"%(torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print('running on: CPU')

# model = Light_Classifier(num_convs_backbone=convs_backbone, num_backbone_out_channels=out_channels_backbone)
model = Light_Classifier()

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

#model.apply(init_weights)
model = model.to(device)
print(model)

transform = transforms.Compose([
    # transforms.Resize([int(IMG_SIZE_X*scale_factor), int(IMG_SIZE_Y*scale_factor)]),
    transforms.ToTensor(), # will put the image range between 0 and 1
])

#load dataset
dataset = Light_Dataset(dataset_path, transform=transform, img_size=n_img_size, dlength=dlength)

# a simple custom collate function, just to show the idea def my_collate(batch):
def my_collate_2(batch):
    imgs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return [imgs, labels]


loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_2)

for i, data in enumerate(loader):
    tic = time()
    imgs, _ = data
    imgs = npasarray(list(nparray(im) for im in imgs))
    # imgs = imgs[0, :, :, :]
    print(imgs.shape)

    # mean0 = npmean(imgs[:,:,:,0])
    # print("m0", mean0)
    # mean1 = npmean(imgs[:,:,:,1])
    # print("m1", mean1)
    # mean2 = npmean(imgs[:,:,:,2])
    # print("m2", mean2)

    # std0 = npstd(imgs[:,:,:,0])
    # print("s0", std0)
    # std1 = npstd(imgs[:,:,:,1])
    # print("s1", std1)
    # std2 = npstd(imgs[:,:,:,2])
    # print("s2", std2)
    print(f"took exactly {(time() - tic)*100} seconds")
    time()
   