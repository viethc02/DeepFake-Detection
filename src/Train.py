import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.Dataset import video_dataset
from model.Model import Model
from torch import nn
from torchvision import models
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
import glob
from albumentations import Compose
from PIL import Image
from torch.autograd import Variable
import time
import sys
import seaborn as sn
from sklearn.metrics import confusion_matrix
import random

def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2 ** 32 if torch_seed >= 0 else (torch_seed + 1) // 2 ** 32 - 1
    random.seed(torch_seed)
    np.random.seed(np_seed)

def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)
    b,g,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
    image = image*255.0
    plt.imshow(image.astype(int))
    plt.show()

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


def train_epoch(epoch, num_epochs, data_loader, model, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    t = []
    for i, (inputs, targets) in enumerate(data_loader):
        #if torch.cuda.is_available():
            #print(targets)
        targets = targets.to(device)
        inputs = inputs.to(device)
        #print(targets)
        _,outputs = model(inputs)
        loss  = criterion(outputs,targets)
        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
                "\r[Epoch %d/%d] [Batch %d / %d] [Loss: %f, Acc: %.2f%%]"
                % (
                    epoch,
                    num_epochs,
                    i,
                    len(data_loader),
                    losses.avg,
                    accuracies.avg))
    torch.save(model.state_dict(),'/kaggle/working/checkpoint.pt')
    return losses.avg,accuracies.avg

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
                    % (
                        i,
                        len(data_loader),
                        loss,
                        acc
                        )
                    )
        print('\nAccuracy {}'.format(accuracies.avg))
    return true,pred,losses.avg,accuracies.avg

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


if __name__ == "__main__":

    label = pd.read_csv('/kaggle/input/my-dfdc/metadata_dfdc.csv')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_video_names_fakes = glob.glob('/kaggle/input/my-deepfake-data/my_train_data/fake_videos/*.mp4')
    train_video_names_reals = glob.glob('/kaggle/input/my-deepfake-data/my_train_data/real_videos/*.mp4')
    train_video_names = train_video_names_fakes + train_video_names_reals
    random.shuffle(train_video_names)
    print(len(train_video_names))

    im_size = 112

    train_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),  
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
                transforms.RandomRotation(degrees=15),  
                transforms.Resize((im_size,im_size)), 
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomRotation(degrees=10, expand=False),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.1),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=7)], p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])# Chuyển ảnh thành tensor
    ])
    train_videos = []
    test_videos = []

    train_videos = train_video_names[:int(0.8*len(train_video_names))]
    test_videos = train_video_names[int(0.8*len(train_video_names)):]

    train_data = video_dataset(video_names=train_videos, labels=label, sequence_length=10, transform=train_transforms)
    val_data = video_dataset(video_names=test_videos, labels=label, sequence_length=10, transform=train_transforms)
    train_loader = DataLoader(train_data, batch_size = 4,shuffle = True,num_workers = 0, worker_init_fn=worker_init_fn)
    valid_loader = DataLoader(val_data, batch_size = 4,shuffle = True,num_workers = 0, worker_init_fn=worker_init_fn)

    model = Model(2).to(device)

    torch.cuda.empty_cache()

    #learning rate
    lr = 1e-5
    num_epochs = 10

    optimizer = torch.optim.Adam(model.parameters(), lr= lr,weight_decay = 1e-5)
    criterion = nn.CrossEntropyLoss().cuda()

    train_loss_avg =[]
    train_accuracy = []
    test_loss_avg = []
    test_accuracy = []

    for epoch in range(1,num_epochs+1):
        l, acc = train_epoch(epoch,num_epochs,train_loader,model,criterion,optimizer)
        train_loss_avg.append(l)
        train_accuracy.append(acc)
        true,pred,tl,t_acc = test(epoch,model,valid_loader,criterion)
        test_loss_avg.append(tl)
        test_accuracy.append(t_acc)

    plot_loss(train_loss_avg,test_loss_avg,len(train_loss_avg))
    plot_accuracy(train_accuracy,test_accuracy,len(train_accuracy))
    print(confusion_matrix(true,pred))
    print_confusion_matrix(true,pred)

    torch.save(model.state_dict(), 'pretrained/Resnet_mydata.pth')