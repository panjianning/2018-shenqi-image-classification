import os
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self, csv_path, data_dir = './', transform=None):
        super().__init__()
        self.df = pd.read_csv(csv_path).values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_name, label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name)
        with Image.open(img_path) as img:
            image = img.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label-1
        
use_gpu = True
num_classes = 5
num_epochs = 30
early_stopping = 10
model = models.densenet201(pretrained=True)

for para in list(model.parameters()):
    para.requires_grad=False
for para in list(model.features.denseblock3.parameters()):
    para.requires_grad=True
for para in list(model.features.transition3.parameters()):
    para.requires_grad=True
for para in list(model.features.denseblock4.parameters()):
    para.requires_grad=True
for para in list(model.features.norm5.parameters()):
    para.requires_grad=True
    
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1920, num_classes)
)

if use_gpu:
    model = model.cuda()
    
model.load_state_dict(torch.load('../input/finetune-densnet201/tuned-densenet201.pth'))

trans_valid = transforms.Compose([transforms.Resize(size=256),
                            transforms.CenterCrop(size=224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

dataset_valid = MyDataset(csv_path='../input/finetune-densnet201/df_valid.csv', 
    data_dir='../input/shenqi/train/data/', transform=trans_valid)

loader_valid = DataLoader(dataset = dataset_valid, batch_size=32, shuffle=False, num_workers=0)

def get_prediction(model, loader, valid=False):
    prediction = np.array([])
    model.eval()
    for _, data in enumerate(loader):
        if valid:
            inputs,_ = data
        else:
            inputs = data
        print('.', end='')
        if use_gpu:
            inputs = inputs.cuda()
        outputs = model(inputs)
        pred = torch.argmax(outputs.data, dim=1)
        prediction = np.append(prediction, pred.cpu().numpy())
    return prediction
    
val_prediction = get_prediction(model, loader_valid, True)
val_true = pd.read_csv('../input/finetune-densnet201/df_valid.csv')[' type'].values-1

val_acc = np.mean(val_prediction==val_true)
from sklearn.metrics import f1_score
val_f1 = 0
for i in range(5):
    val_f1 += f1_score(val_prediction==i, val_true==i)
val_f1 = val_f1/5
print('val acc: %.6f, val f1: %.6f' % (val_acc, val_f1))

import cv2

class TestDataset(Dataset):
    def __init__(self, data_dir = './', transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.image_names = os.listdir(data_dir)
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        img_name = self.image_names[index]
        img_path = os.path.join(self.data_dir, img_name)
        try:
            img = Image.open(img_path)
        except OSError:
            print('read with cv2')
            img = Image.fromarray(cv2.imread(img_path))
        image = img.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

dataset_test = TestDataset(data_dir='../input/shenqi/test/', transform=trans_valid)
loader_test = DataLoader(dataset = dataset_test, batch_size=32, shuffle=False, num_workers=0)

test_prediction = get_prediction(model, loader_test)
sub = pd.DataFrame(list(zip(dataset_test.image_names,test_prediction.astype(int)+1)),
                   columns=['filename', ' type'])
sub.to_csv('pytorch-densenet201.csv',index=False)
