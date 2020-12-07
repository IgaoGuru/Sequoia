import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2

class Light_Classifier(torch.nn.Module):
   def __init__(self):
      super(Light_Classifier, self).__init__()
      self.conv1 = self.conv_block(c_in = 3, c_out = 18, kernel_size = 3, stride = 1, padding = 1)
      self.conv2 = self.conv_block(c_in = 3, c_out = 18, kernel_size = 3, stride = 1, padding = 1)
      # self.fc1 = nn.Linear(18 * 16 * 16, 64)
      # self.fc2 = nn.Linear(64, 10)

   def conv_block(self, c_in, c_out, dropout=0.1, kernel_size=3, stride=1, **kwargs):
      seq_block = nn.Sequential(
         nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
         nn.BatchNorm2d(num_features=c_out),
         nn.ReLU(),
         nn.Dropout2d(p=dropout)
      )
      return seq_block

   def forward(self, x):
      x = self.conv1(x)
      x = self.maxpool(x)
      print(x.shape)

      x = self.conv2(x)
      x = self.maxpool(x)

      return x
      
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)    

   correct_results_sum = (y_pred_tags == y_test).sum().float() 

   acc = correct_results_sum/y_test.shape[0]
   acc = torch.round(acc * 100)    
   return acc