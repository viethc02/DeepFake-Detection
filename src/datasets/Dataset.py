import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch import nn
from torchvision import models
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


class video_dataset(Dataset):
    def __init__(self, video_names, labels, sequence_length = 32,transform = None):
        self.video_names = video_names
        self.labels = labels
        self.transform = transform
        self.count = sequence_length
    def __len__(self):

        return len(self.video_names)
    def __getitem__(self, id):
        frames = []
        video_path = self.video_names[id]
        temp_video = video_path.split('/')[-2]
#         if '(1)' in temp_video:
#             temp_video = temp_video.replace("(1)","")
#         if not self.labels.loc[self.labels["URI"] == temp_video].empty:
#             index = self.labels.loc[self.labels["URI"] == temp_video].index.values[0]
#             label = self.labels.iloc[index, 1]
#         else:
#             print(temp_video)
#             return
    # if 'origin' in temp_video:
    #   label = 0
    # elif 'Deepfakes' in temp_video:
    #   label = 1
    # elif 'Face2Face' in temp_video:
    #   label = 2
    # elif 'FaceShifter' in temp_video:
    #   label = 3
    # elif 'FaceSwap' in temp_video:
    #   label = 4
    # elif 'NeuralTextures' in temp_video:
    #   label = 5
        if 'fake' in temp_video:
            label = 0
        else:
            label = 1
#         if(label == 'FAKE'):
#             label = 0
#         if(label == 'REAL'):
#             label = 1

        for i,frame in enumerate(self.frame_extract(video_path)):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        #print(len(frames))
        if len(frames) == 0:
            return
        if (len(frames) > 0 and len(frames) < self.count):
            for i in range(self.count - len(frames)):
                frames.append(frames[-1])
        # print(padded_tensors)
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames, label
    def frame_extract(self,path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image