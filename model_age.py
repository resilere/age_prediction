# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:27:25 2021

@author: eser
"""
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
class Net2_5D(nn.Module):
    
    '''this is 2,5 D triplanar CNN model'''
    
    def __init__(self):
        super(Net2_5D, self).__init__()
        self.conv0 = nn.Conv2d(256, 128, 5, padding = 2 )
        self.conv1 = nn.Conv2d(128, 64, 5, padding=2)
        
        self.conv2 = nn.Conv2d(64, 32, 5, padding=2)
        
        self.conv3 = nn.Conv2d(32, 16, 5, padding=2)
        
        
        
        self.conv4 = nn.Conv2d(16, 8, 5, padding =2 )
               
        
        
        self.fc1 = nn.Linear(8*16*16,  8*8*8)
        self.fc2 = nn.Linear( 8*8*8, 8*4*4)
        self.fc3 = nn.Linear(8*4*4, 1)
        
        
        """this is batch normalization, for 32 channels, implemented after 
        convolutional layers but before Relu, except the last layer"""
        self.m128 = nn.BatchNorm2d(128)
        self.m64 = nn.BatchNorm2d(64)
        self.m32 = nn.BatchNorm2d(32)
        self.m16 = nn.BatchNorm2d(16)
        self.m8 = nn.BatchNorm2d(8)
        #self.m32 = nn.BatchNorm2d(32)
    def forward(self, x):
        """Here we take the permutations of the dimensions of the input patch and pass through CNN layers"""
        planes = []
        
# =============================================================================
#         y = z = x
#         
#         y = y.permute(0, 2, 3, 1)
#         z = z.permute(0, 3, 1, 2)
#         planes.append(y)
#         planes.append(z)
#         
# =============================================================================
        planes.append(x)
        
        
        concat = []
        '''this is a loop to implement the same cnn layers for the different planes of the 3d patch'''
        for x in planes:
            x = F.relu(self.m128(self.conv0(x)))
            x = F.relu(self.m64(self.conv1(x)))
            
            
            x = F.max_pool2d(x, (2,2))
            
            x = F.relu(self.m32(self.conv2(x)))
            
            
            x = F.max_pool2d(x, (2,2))
            
            x = F.relu(self.m16(self.conv3(x)))
            
            x = F.max_pool2d(x, (2,2))         
            
            x = F.relu(self.m8(self.conv4(x)))
            
            x = F.max_pool2d(x, (2,2)) 
            
            x = x.view(-1, 8 * 16* 16)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
           

        return x
class Net_new(nn.Module):
    
    '''this is a new CNN model for 3d ct images'''
    
    def __init__(self):
        super(Net_new, self).__init__()
        self.conv0 = nn.Conv2d(256, 512, 5, padding = 2 )
        self.conv1 = nn.Conv2d(512, 128, 5, padding=2)
        
        self.conv2 = nn.Conv2d(128,64, 5, padding=2)
        
        self.conv3 = nn.Conv2d(64, 32 , 5, padding=2)
        
        
        
        self.conv4 = nn.Conv2d(32, 16, 5, padding =2 )
               
        
        
        self.fc1 = nn.Linear(16*16*16, 8*8*8)
        self.fc2 = nn.Linear(8*8*8,4*4*4)
        self.fc3 = nn.Linear(4*4*4, 1)
        self.dropout = nn.Dropout(0.25)
        
        """this is batch normalization, for 32 channels, implemented after 
        convolutional layers but before Relu, except the last layer"""
        self.m512 = nn.BatchNorm2d(512)
        self.m128 = nn.BatchNorm2d(128)
        self.m64 = nn.BatchNorm2d(64)
        
        self.m16 = nn.BatchNorm2d(16)
        #self.m8 = nn.BatchNorm2d(8)
        self.m32 = nn.BatchNorm2d(32)
    def forward(self, x):
        """Here we take the permutations of the dimensions of the input patch and pass through CNN layers"""
        planes = []
        
# =============================================================================
#         y = z = x
#         
#         y = y.permute(0, 2, 3, 1)
#         z = z.permute(0, 3, 1, 2)
#         planes.append(y)
#         planes.append(z)
#         
# =============================================================================
        planes.append(x)
        
        
        '''this is a loop to implement the same cnn layers for the different planes of the 3d patch'''
        for x in planes:
            x = F.relu(self.m512(self.conv0(x)))
            x = F.max_pool2d(x, (2,2))
            x = self.dropout(x)
            x = F.relu(self.m128(self.conv1(x)))
            
            
            x = F.max_pool2d(x, (2,2))
            x = self.dropout(x)
            x = F.relu(self.m64(self.conv2(x)))
            
            
            x = F.max_pool2d(x, (2,2))
            x = self.dropout(x)
            x = F.relu(self.m32(self.conv3(x))) 
            x = self.dropout(x)
            x = F.max_pool2d(x, (2,2)) 
            x = F.relu(self.m16(self.conv4(x))) 
            
            x = x.view(-1, 16*16*16)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
           

        return x
class CT_ages(Dataset):
    """Cardiac ctimages with ages dataset."""

    def __init__(self, images_dir, labels_dir,transform = None):
        """
        Args:
            labels= array that has the labels
            images= arrays including the [256,512,512] array of images
        """
        self.labels= sorted(os.listdir(labels_dir))
        self.images = sorted(os.listdir(images_dir))
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images_dir + '\\' + self.images[idx]
        print('index',idx)
        msk_path = self.labels_dir + '\\' + self.labels[idx]
        
        image = np.load(img_path)
        
        label = np.load(msk_path)
        
        
        
        sample = {'image': image, 'label': label}
        if self.transform:
            sample['image'] = self.transform(torch.tensor(sample['image']))
        
        return sample
    
class DiceLoss(nn.Module):
    """this is the working dice loss  fuction from kaggle"""
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth= 0.000001):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.softmax(inputs, dim = 1)       

        inputs1 = inputs[: , 1, : , : , : ]
        targets1 = targets[: , 1, : , : , : ]
        inputs2 = inputs[:,0, : , : , : ]
        targets2 = targets[:,0, : , : , : ]

      
        #import pdb;pdb.set_trace()
        #flatten label and prediction tensors
        inputs1 = inputs1.view(-1)
        targets1 = targets1.reshape(-1) #Ich habe hier von view geandert weil es ein error gibt
        
        intersection1 = (inputs1 * targets1).sum()                            
        dice1 = (2.*intersection1 + smooth)/((inputs1**2).sum() + (targets1**2).sum() + smooth) 
        inputs2 = inputs2.view(-1)
        targets2 = targets2.reshape(-1) #Ich habe hier von view geandert weil es ein error gibt
        
        intersection2 = (inputs2 * targets2).sum()                            
        dice2 = (2.*intersection2 + smooth)/((inputs2**2).sum() + (targets2**2).sum() + smooth) 
        #print("dice1, dice2", dice1, dice2)
        return 1 - dice1 -0.3*dice2