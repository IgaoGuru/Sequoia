import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset

import numpy as np
import cv2
from time import sleep
from random import randint
from tqdm import tqdm
import os
from PIL import Image

from datasetcsgo import CsgoDataset
# import matplotlib.pyplot as plt
# import pandas as pd

class Light_Classifier(torch.nn.Module):
   def __init__(self):
      super(Light_Classifier, self).__init__()
      self.conv1 = self.conv_block(c_in = 3, c_out = 15, kernel_size = 3, stride = 1, padding = 1)
      self.conv2 = self.conv_block(c_in = 15, c_out = 12, kernel_size = 3, stride = 1, padding = 1)
      self.conv3 = self.conv_block(c_in = 12, c_out = 3, kernel_size = 3, stride = 1, padding = 1)
      # 32px --> 48
      # 100px --> 432
      # self.bigN = 432
      self.bigN = 48
      self.fc1 = nn.Linear(self.bigN, 32)
      self.fc2 = nn.Linear(32, 16)
      self.fc3 = nn.Linear(16, 1)
      self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
      # self.prefc1 = nn.Linear(self.bigN, )

   def conv_block(self, c_in, c_out, dropout=0.1, kernel_size=3, stride=1, **kwargs):
      seq_block = nn.Sequential(
         nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, **kwargs),
         nn.BatchNorm2d(num_features=c_out),
         nn.ReLU(),
         # nn.Dropout2d(p=dropout)
      )
      return seq_block

   def forward(self, x):
      x = self.conv1(x)
      x = self.maxpool(x)

      x = self.conv2(x)
      x = self.maxpool(x)

      x = self.conv3(x)
      x = self.maxpool(x)
      x = x.reshape((-1, self.bigN))

      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = torch.sigmoid(self.fc3(x))
      return x

class Heavy_Classifier(torch.nn.Module):
   def __init__(self):
      super(Heavy_Classifier, self).__init__()
      self.conv1 = self.conv_block(c_in = 3, c_out = 25, kernel_size = 7, stride = 1, padding = 1)
      self.conv2 = self.conv_block(c_in = 25, c_out = 40, kernel_size = 5, stride = 1, padding = 1)
      self.conv3 = self.conv_block(c_in = 40, c_out = 40, kernel_size = 5, stride = 1, padding = 1)
      self.conv4 = self.conv_block(c_in = 40, c_out = 25, kernel_size = 3, stride = 1, padding = 1)
      self.conv5 = self.conv_block(c_in = 25, c_out = 25, kernel_size = 3, stride = 1, padding = 1)
      self.conv6 = self.conv_block(c_in = 25, c_out = 3, kernel_size = 3, stride = 1, padding = 1)
      # 28px --> ???
      # 32px --> 3
      # 100px --> 48 (heavy)
      self.bigN = 3
      self.fc1 = nn.Linear(self.bigN, 64)
      self.fc2 = nn.Linear(64, 9)
      self.fc3 = nn.Linear(9, 2)
      self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

   def conv_block(self, c_in, c_out, dropout=0.1, kernel_size=3, stride=1, **kwargs):
      seq_block = nn.Sequential(
         nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, **kwargs),
         nn.BatchNorm2d(num_features=c_out),
         nn.ReLU(),
         nn.Dropout2d(p=dropout)
      )
      return seq_block

   def forward(self, x):
      x = self.conv1(x)
      x = self.maxpool(x)
      x = self.conv2(x)
      x = self.conv3(x)
      x = self.maxpool(x)
      x = self.conv4(x)
      x = self.conv5(x)
      x = self.maxpool(x)
      x = self.conv6(x)
      x = self.maxpool(x)
      x = x.reshape((-1, self.bigN))

      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      # x = F.relu(x)
      # x = torch.sigmoid(self.fc3(x))
      return x

class Ultra_Light_Classifier(torch.nn.Module):
   def __init__(self):
      super(Ultra_Light_Classifier, self).__init__()
      self.conv1 = self.conv_block(c_in = 3, c_out = 64, kernel_size = 3, stride = 1, padding = 2, dropout=0.1)
      self.conv2 = self.conv_block(c_in = 64, c_out = 32, kernel_size = 3, stride = 1, padding = 2, dropout=0.1)
      self.conv3 = self.conv_block(c_in = 32, c_out = 16, kernel_size = 3, stride = 1, padding = 2, dropout=0.1)
      self.lastcnn = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=0)
      # 32px --> 3840
      self.bigN = 3840
      self.fc1 = nn.Linear(self.bigN, 1024)
      self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
      # self.prefc1 = nn.Linear(self.bigN, )

   def conv_block(self, c_in, c_out, dropout=0.1, kernel_size=3, stride=1, **kwargs):
      seq_block = nn.Sequential(
         nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, **kwargs),
         nn.BatchNorm2d(num_features=c_out),
         nn.ReLU(),
         # nn.Dropout2d(p=dropout)
      )
      return seq_block

   def forward(self, x):

      x = self.conv1(x)
      x = self.maxpool(x)

      x = self.conv2(x)

      x = self.conv3(x)
      x = self.maxpool(x)

      x = self.lastcnn(x)

      return x


class Light_Dataset(CsgoDataset):
   def __init__(self, root_path, img_size=100, transform=None, scale_factor=None, dlength=None):
      print('building dataset! please wait a moment')
      super().__init__(root_path, dlength=dlength)
      self.img_size = img_size

   #crop image on __getitem__
   def __getitem__(self, idx):
      img, bboxes, labels = super().__getitem__(idx)
      bbox = bboxes[0]
      label = labels[0]

      img = img.crop(bbox.tolist())
      img = img.resize((self.img_size, self.img_size))

      if self.transform:
         img = self.transform(img)

      return img, label

#some utils      
def binary_acc(y_pred, y_test):
   y_pred_tag = torch.log_softmax(y_pred, dim = 1)
   _, y_pred_tags = torch.max(y_pred_tag, dim = 1)    
   correct_results_sum = (y_pred_tags == y_test).sum().float() 

   acc = correct_results_sum/y_test.shape[0]
   acc = torch.round(acc * 100)    
   return acc

# root_path = "e:/documento/outputs/"
# dset = Light_Dataset(root_path, img_size=32, dlength=4000)
# ct = 0
# tr = 0
# for i in tqdm(range(4000)):
#    # -- for checking image quality ---
#    u = randint(1, 4000)
#    sample = dset[u]
#    img = sample[0]
#    label = sample[1].item()
#    img = img.resize([600,600])
#    img = np.array(img)
#    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#    cv2.imshow(f"igor - {label}", img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
   
#    --for checking parity ---
#    sample = dset[u]
#    _, label = sample
#    if label.item() == 0:
#       ct += 1
#    else:
#       tr += 1
# print(f"ct {ct}, tr: {tr}")