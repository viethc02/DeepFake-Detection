import torch
import torchvision
from torch import nn
from torchvision import models
import os
import numpy as np
import cv2
from PIL import Image

import timm
class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        #model = models.efficientnet_b7(pretrained = False) #Residual Network CNN
        #model = timm.create_model('xception', pretrained=True)
        #model = models.vgg19(pretrained = True)
        # model = models.wide_resnet50_2(pretrained = False)
        model = models.resnext50_32x4d(pretrained = True)
        #model = timm.create_model('swin_large_patch4_window7_224.ms_in22k', pretrained=False)
        #model = timm.create_model("vit_base_patch16_224", pretrained=False)
        #model = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

        #model = models.resnet50(pretrained = False)
        self.model = nn.Sequential(*list(model.children())[:-2])
        #self.model = model
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x_resized_1 = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x_resized_1)
        #print(x_resized_1.shape)
        x_resized_1 = self.avgpool(fmap)
        #print(x_resized_1.shape)
        # print(batch_size, seq_length)
        x_resized_2 = x_resized_1.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x_resized_2,None)
        return fmap,self.dp(self.linear1(torch.mean(x_lstm,dim = 1)))
        #return _, self.linear1(x_resized_2)