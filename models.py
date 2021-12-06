import os
import torch
import torch.nn as nn
import torchvision.models
import math
import torch.nn.functional as F

def convlayer(in_channels,out_channels,kernel_size,stride):
    padding = 1
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU6(inplace=True),
    )

def mobilenetlayer(in_channels,out_channels,kernel_size,stride):
  padding = 1
  return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=stride,padding=padding,bias=False,groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.ReLU6(inplace=True),
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU6(inplace=True),
    )
def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
    )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
    )

class Encoder(nn.Module):
  def __init__(self):
        super(Encoder, self).__init__()
        kernel_size = 3
        self.conv0 = convlayer(3,16,kernel_size,2)
        self.conv1 = mobilenetlayer(16,56,kernel_size,1)
        self.conv2 = mobilenetlayer(56,88,kernel_size,2)
        self.conv3 = mobilenetlayer(88,120,kernel_size,1)
        self.conv4 = mobilenetlayer(120,144,kernel_size,2)
        self.conv5 = mobilenetlayer(144,256,kernel_size,1)
        self.conv6 = mobilenetlayer(256,408,kernel_size,2)
        self.conv7 = mobilenetlayer(408,376,kernel_size,1)
        self.conv8 = mobilenetlayer(376,272,kernel_size,1)
        self.conv9 = mobilenetlayer(272,288,kernel_size,1)
        self.conv10 = mobilenetlayer(288,296,kernel_size,1)
        self.conv11 = mobilenetlayer(296,328,kernel_size,1)
        self.conv12 = mobilenetlayer(328,480,kernel_size,2)
        self.conv13 = mobilenetlayer(480,512,kernel_size,1)
  def forward(self, x):
        for i in range(14):
            layer = getattr(self, 'conv{}'.format(i))
            x = layer(x)
            if i==1:
                x1 = x
            elif i==3:
                x2 = x
            elif i==5:
                x3 = x
        return x,x1,x2,x3
class Decoder(nn.Module):
  def __init__(self):
        super(Decoder, self).__init__()
        kernel_size = 5
        self.decode_conv1 = nn.Sequential(
            depthwise(512, kernel_size),
            pointwise(512, 200))
        self.decode_conv2 = nn.Sequential(
            depthwise(200, kernel_size),
            pointwise(200, 256))
        self.decode_conv3 = nn.Sequential(
            depthwise(256, kernel_size),
            pointwise(256, 120))
        self.decode_conv4 = nn.Sequential(
            depthwise(120, kernel_size),
            pointwise(120, 56))
        self.decode_conv5 = nn.Sequential(
            depthwise(56, kernel_size),
            pointwise(56, 16))
        self.decode_conv6 = pointwise(16, 1)
  def forward(self, x,x1,x2,x3):
        for i in range(1,6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i==4:
                x = x + x1
            elif i==3:
                x = x + x2
            elif i==2:
                x = x + x3
        x = self.decode_conv6(x)
        return x

class DepthEstimation(nn.Module):
  def __init__(self):
        super(DepthEstimation, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
  def forward(self,x):
        x,x1,x2,x3 = self.encoder(x)
        x = self.decoder(x,x1,x2,x3)
        return x
