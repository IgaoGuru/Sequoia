import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset

import numpy as np
import cv2
from time import sleep
import os
from PIL import Image

from datasetcsgo import CsgoDataset
# import matplotlib.pyplot as plt
# import pandas as pd

class Light_Classifier(torch.nn.Module):
   def __init__(self):
      super(Light_Classifier, self).__init__()
      self.conv1 = self.conv_block(c_in = 3, c_out = 12, kernel_size = 3, stride = 1, padding = 1)
      self.conv2 = self.conv_block(c_in = 12, c_out = 3, kernel_size = 3, stride = 1, padding = 1)
      self.fc1 = nn.Linear(147, 2)
      self.fc2 = nn.Linear(64, 10)
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
      x = self.maxpool(x)
      x = x.reshape((-1, 147))

      x = self.fc1(x)
      return x

class Light_Dataset(CsgoDataset):
   def __init__(self, root_path, img_size=1000, transform=None, scale_factor=None, dlength=None):
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

      return img, label

      # img = self.dset.get_image(idx)
      # if self.transform:
      #    img = self.transform(img)
      # #clazz = self.dict_frames[self.frame_keys[idx]][0]

      # bboxes = torch.tensor(self.dset.dict_dataset[self.dset.frame_keys[idx]][1], dtype=torch.float)
      # if self.scale_factor != None:
      #    bboxes = bboxes * self.scale_factor
      # labels = torch.tensor([self.dset.classes.index(c) for c in self.dset.dict_dataset[self.dset.frame_keys[idx]][0]], dtype=torch.int64)
      # return img, bboxes, labels

#some utils      
def binary_acc(y_pred, y_test):
   y_pred_tag = torch.log_softmax(y_pred, dim = 1)
   _, y_pred_tags = torch.max(y_pred_tag, dim = 1)    
   correct_results_sum = (y_pred_tags == y_test).sum().float() 

   acc = correct_results_sum/y_test.shape[0]
   acc = torch.round(acc * 100)    
   return acc

# root_path = "e:/documento/outputs/"
# dset = Light_Dataset(root_path, img_size=28, dlength=5000)
# for i in range(50):
#    sample = dset[i]

#    img = sample[0]
#    img = np.array(img)
#    print(img)
#    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#    cv2.imshow("igor", img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()