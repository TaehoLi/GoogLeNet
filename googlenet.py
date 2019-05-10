import os
import torch
import torch.nn as nn
import torchvision

import numpy as np
import cv2
import time
import copy

class InceptionBlock(nn.Module):
    """
    Inception version
    naive : without dimension reduction
    reduce : with dimension reduction
    """
    def __init__(self, in_=0, out_1x1=0, mid_3x3=0, out_3x3=0,
                 mid_5x5=0, out_5x5=0, pool_out=0, version='reduce'):
        super(InceptionBlock, self).__init__()
        assert version in ['reduce', 'naive']
        
        if version == 'naive':
            self.conv_1x1 = nn.Sequential(
                nn.Conv2d(in_channels=in_, out_channels=out_1x1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.LeakyReLU(inplace=True)
            )
            self.conv_3x3 = nn.Sequential(
                nn.Conv2d(in_channels=in_, out_channels=out_3x3, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(inplace=True)
            )
            self.conv_5x5 = nn.Sequential(
                nn.Conv2d(in_channels=in_, out_channels=out_5x5, kernel_size=5, stride=1, padding=2, bias=True),
                nn.LeakyReLU(inplace=True)
            )
            self.maxpool_3x3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            
        elif version == 'reduce':
            self.conv_1x1 = nn.Sequential(
                nn.Conv2d(in_channels=in_, out_channels=out_1x1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.LeakyReLU(inplace=True)
            )
            self.conv_3x3 = nn.Sequential(
                nn.Conv2d(in_channels=in_, out_channels=mid_3x3, kernel_size=1, stride=1, padding=0, bias=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=mid_3x3, out_channels=out_3x3, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(inplace=True)
            )
            self.conv_5x5 = nn.Sequential(
                nn.Conv2d(in_channels=in_, out_channels=mid_5x5, kernel_size=1, stride=1, padding=0, bias=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=mid_5x5, out_channels=out_5x5, kernel_size=5, stride=1, padding=2, bias=True),
                nn.LeakyReLU(inplace=True)
            )
            self.maxpool_3x3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=in_, out_channels=pool_out, kernel_size=1, stride=1, padding=0, bias=True),
                nn.LeakyReLU(inplace=True)
            )
        
    def forward(self, input):
        feature_1x1 = self.conv_1x1(input)
        feature_3x3 = self.conv_3x3(input)
        feature_5x5 = self.conv_5x5(input)
        feature_pool = self.maxpool_3x3(input)
        
        feature = torch.cat([feature_1x1, feature_3x3, feature_5x5, feature_pool],dim=1 )
        
        return feature
    
class _AuxiliaryBlock(nn.Module):
    def __init__(self, in_channel, num_classes=1):
        super(_AuxiliaryBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=128, eps=0.001),
            nn.ReLU(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((4,4))
        
        self.linear = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, input):
        input = self.avgpool(input)
        input = self.conv(input)
        # Nx128x4x4
        input = input.view(input.size(0), -1)
        # Nx2048
        input = self.linear(input)
        # Nxnum_classes
        
        return input
          
        
class GoogLeNet(nn.Module):
    def __init__(self, in_channel=1, num_classes=1, aux_block=True):
        super(GoogLeNet, self).__init__()
        self.aux_block = aux_block
        
        # conv block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64, eps=0.001),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # conv block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=64, eps=0.001),
            nn.ReLU(inplace=True)
        )
        # conv block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        
        if aux_block:
            self.aux_block1 = _AuxiliaryBlock(512, num_classes)
            self.aux_block2 = _AuxiliaryBlock(528, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout2d(0.4)
        self.weight_softmax = nn.Linear(1024, num_classes)
        
    def forward(self, input):
        input = self.conv_block1(input)
        print("conv_block1 shape : ", input.shape)
        input = self.maxpool1(input)
        input = self.conv_block2(input)
        input = self.conv_block3(input)
        input = self.maxpool2(input)
        print("maxpool2 shape : ", input.shape)
        input = self.inception3a(input)
        input = self.inception3b(input)
        input = self.maxpool3(input)
        
        input = self.inception4a(input)
        # N x 512
        if self.training and self.aux_block:
            aux1 = self.aux_block1(input)
            
        input = self.inception4b(input)
        input = self.inception4c(input)
        input = self.inception4d(input)
        # N x 528
        if self.training and self.aux_block:
            aux2 = self.aux_block2(input)
        
        input = self.inception4e(input)
        input = self.maxpool4(input)
        print("maxpool4 shape : ", input.shape)
        print("___________________________")
        input = self.inception5a(input)
        # N x 832
        input = self.inception5b(input)
        # N x 1024
        
        input = self.avgpool(input)
        input = input.view(input.size(0), -1)
        input = self.dropout(input)
        input = self.weight_softmax(input)
        
        if self.training and self.aux_block:
            return aux1.reshape((-1)), aux2.reshape((-1)), input.reshape((-1))
        return input.reshape((-1))
        