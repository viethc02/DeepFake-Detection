# -*- coding: utf-8 -*-
"""Predict.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VAEMWCxGJrxVM0hyJGITaTo7JTw0Fpob
"""

# from google.colab import drive
# drive.mount('/content/drive')

# import os
# os.chdir('drive/MyDrive/DeepFake')

#import libraries
# !pip3 install face_recognition
# !pip install timm

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import timm
#import face_recognition

#import libraries
import torch
from torch.autograd import Variable
import time
import os
import sys
import os
from torch import nn
from torchvision import models
import glob

#Model with feature visualization
from torch import nn
from torchvision import models
class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 512, lstm_layers=1 , hidden_dim = 512, bidirectional = False):
        super(Model, self).__init__()
        model = models.vgg19(pretrained = False)
        #model = timm.create_model('xception', pretrained=False)
        #model = models.efficientnet_b7(pretrained = False)
        #model = models.resnext50_32x4d(pretrained=False)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(512,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,512)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))

# im_size = 112
# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]
# sm = nn.Softmax()
# inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
# def im_convert(tensor):
#     """ Display a tensor as an image. """
#     image = tensor.to("cpu").clone().detach()
#     image = image.squeeze()
#     image = inv_normalize(image)
#     image = image.numpy()
#     image = image.transpose(1,2,0)
#     image = image.clip(0, 1)
#     cv2.imwrite('./2.png',image*255)
#     return image

# def predict(model,img,path = './'):
#   fmap,logits = model(img.to(device))
#   params = list(model.parameters())
#   weight_softmax = model.linear1.weight.detach().cpu().numpy()
#   logits = sm(logits)
#   _,prediction = torch.max(logits,1)
#   confidence = logits[:,int(prediction.item())].item()*100
#   print('confidence of prediction:',logits[:,int(prediction.item())].item()*100)
#   idx = np.argmax(logits.detach().cpu().numpy())
#   bz, nc, h, w = fmap.shape
#   out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
#   predict = out.reshape(h,w)
#   predict = predict - np.min(predict)
#   predict_img = predict / np.max(predict)
#   predict_img = np.uint8(255*predict_img)
#   out = cv2.resize(predict_img, (im_size,im_size))
#   heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
#   img = im_convert(img[:,-1,:,:,:])
#   result = heatmap * 0.5 + img*0.8*255
#   cv2.imwrite('/content/1.png',result)
#   result1 = heatmap * 0.5/255 + img*0.8
#   r,g,b = cv2.split(result1)
#   result1 = cv2.merge((r,g,b))
#   plt.imshow(result1)
#   plt.show()
#   return [int(prediction.item()),confidence]
# #img = train_data[100][0].unsqueeze(0)
# #predict(model,img)

#!pip3 install face_recognition
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
#import face_recognition
class validation_dataset(Dataset):
    def __init__(self,video_names,sequence_length = 32,transform = None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
    def __len__(self):
        return len(self.video_names)
    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0,a)
        for i,frame in enumerate(self.frame_extract(video_path)):
            #if(i % a == first_frame):
            # faces = face_recognition.face_locations(frame)
            # try:
            #   top,right,bottom,left = faces[0]
            #   frame = frame[top:bottom,left:right,:]
            # except:
            #   pass
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
              break
        #print("no of frames",len(frames))
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path)
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image
def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)
    b,g,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
    image = image*255.0
    plt.imshow(image.astype(int))
    plt.show()

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
        temp_video = video_path.split('/')[-1]
        if '(1)' in temp_video:
            temp_video = temp_video.replace("(1)","")
        if 'FAKE_' in temp_video:
            temp_video = temp_video.replace("FAKE_","")
        if 'REAL_' in temp_video:
            temp_video = temp_video.replace("REAL_","")
        temp_video = temp_video.replace(" ","")
        if not self.labels.loc[self.labels["filename"] == temp_video].empty:
            index = self.labels.loc[self.labels["filename"] == temp_video].index.values[0]
            label = self.labels.iloc[index, 1]
        # Tiếp tục xử lý với giá trị 'value' đã lấy được
        else:
            print(temp_video)
            return
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
#         if 'origin' in temp_video:
#             label = 1
#         else:
#             label = 0
        # if(label == 'FAKE'):
        #     label = 0
        # if(label == 'REAL'):
        #     label = 1

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
#         print(label)
        return frames, label
    def frame_extract(self,path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import pandas as pd
label = pd.read_csv('data/test_video/test.csv')

#Code for making prediction
im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        #transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        #transforms.Normalize(mean,std)
                                        ])

#path_to_videos= ["/content/drive/My Drive/Colab Notebooks/DeepFake/data/test_video_cropped/aqrsylrzgi.mp4"]

path_to_videos = glob.glob('data/test_video_cropped/*.mp4')

test_dataset = video_dataset(path_to_videos, labels=label, sequence_length = 20,transform = train_transforms)
valid_loader_FF = DataLoader(test_dataset, batch_size = 1,shuffle = True,num_workers = 0)

model = Model(2).to(device)
path_to_model = 'pre_trained/vgg19_DFDC.pth'
model.load_state_dict(torch.load(path_to_model))
model.eval()
print('ok')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return 100* n_correct_elems / batch_size

# Commented out IPython magic to ensure Python compatibility.
def test(epoch,model, data_loader ,criterion):
    print('Testing')
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pred = []
    true = []
    count = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            #if torch.cuda.is_available():
            targets = targets.to(device)
            inputs = inputs.to(device)
            _,outputs = model(inputs)
            loss = torch.mean(criterion(outputs, targets))
            acc = calculate_accuracy(outputs,targets)
            _,p = torch.max(outputs,1)
            true += (targets).detach().cpu().numpy().reshape(len(targets)).tolist()
            pred += p.detach().cpu().numpy().reshape(len(p)).tolist()
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            print(
                    "\r[Batch %d / %d]  [Loss: %f, Acc: %.2f%%]"
#                     % (
                        i,
                        len(data_loader),
                        losses.avg,
                        accuracies.avg
                        )
                    )
        print('\nAccuracy {}'.format(accuracies.avg))
    return true,pred,losses.avg,accuracies.avg

import seaborn as sn
from sklearn.metrics import confusion_matrix
#Output confusion matrix
def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print('True positive = ', cm[0][0])
    print('False positive = ', cm[0][1])
    print('False negative = ', cm[1][0])
    print('True negative = ', cm[1][1])
    print('\n')
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.ylabel('Actual label', size = 20)
    plt.xlabel('Predicted label', size = 20)
    plt.xticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.yticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.ylim([2, 0])
    plt.show()
    calculated_acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+ cm[1][1])
    print("Calculated Accuracy",calculated_acc*100)

def plot_loss(train_loss_avg,test_loss_avg,num_epochs):
  loss_train = train_loss_avg
  loss_val = test_loss_avg
  print(num_epochs)
  epochs = range(1,num_epochs+1)
  list_epochs = list(epochs)
  print(list_epochs)
  plt.plot(list_epochs, loss_train, 'g', label='Training loss')
  plt.plot(list_epochs, loss_val, 'b', label='validation loss')
  plt.title('Training and Validation loss')
  plt.xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()
def plot_accuracy(train_accuracy,test_accuracy,num_epochs):
  loss_train = train_accuracy
  loss_val = test_accuracy
  epochs = range(1,num_epochs+1)
  list_epochs = list(epochs)
  print(list_epochs)
  plt.plot(list_epochs, loss_train, 'g', label='Training accuracy')
  plt.plot(list_epochs, loss_val, 'b', label='validation accuracy')
  plt.title('Training and Validation accuracy')
  plt.xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()

criterion = nn.CrossEntropyLoss().cuda()
true,pred,tl,t_acc = test(1,model,valid_loader_FF,criterion)

print(confusion_matrix(true,pred))
print_confusion_matrix(true,pred)

