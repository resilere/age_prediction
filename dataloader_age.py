# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:00:04 2021

@author: eser
"""
import model_age as model
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
directory_image= r'G:\codes\images'

directory_label= r'G:\codes\labels_age'


batch_size = 1
validation_split = .2
shuffle_dataset = True
random_seed= 42
N_EPOCH =10
OUTPUT_FREQUENCY = 5
MIN_LOSS = 50
PATH = r'G:\codes\age_model_new.pth'

transform_train = T.Compose([
    T.CenterCrop(256),
    T.RandomHorizontalFlip(),
    
    
])
dataset=model.CT_ages(directory_image,directory_label,transform = transform_train)

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]


train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)



train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

net = model.Net_new()
net.load_state_dict(torch.load(PATH))
net.train()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)

for epoch in range(N_EPOCH):  # loop over the dataset multiple times

    train_loss = 0.0
    valid_loss = 0.0
    correct = 0
    total = 0
    for i, sample in enumerate(train_loader, 0):
        
        """ get the inputs; data is a list of [inputs, labels] """
        net.train()
        input_image = sample["image"].float()
        label = sample["label"].float()
        print('label',label)
        
        '''zero the parameter gradients'''
        
        optimizer.zero_grad()
        ''' here the model is used and viewed as 5 dimensional tensor '''
        output_class = net(input_image)
        
        #import pdb; pdb.set_trace()
        print('output',output_class)
        #preds = torch.argmax(output_class, dim=1)
        #print(preds,label)
        loss= criterion(output_class,label)
        
        loss.backward()
        
        #list_of_conv = [net.conv0, net.conv1, net.conv2, net.conv3, net.conv4, net.conv5, net.conv7] 
        
        optimizer.step()
        print('round', torch.round(output_class),label,(label-6 < torch.round(output_class) and torch.round(output_class) < label+6) )
        if label-6 < torch.round(output_class) and torch.round(output_class) < label+6:
            correct += 1
        print('correct',correct)
        '''print statistics'''
        train_loss += loss.item()
        total += 1
        if i % OUTPUT_FREQUENCY == OUTPUT_FREQUENCY - 1:    
            """# print every OUTPUT_FREQUENCY mini-batches"""
            plt.clf()
            print('[%d, %5d] train loss: %.3f' %
                  (epoch + 1, i + 1, train_loss / OUTPUT_FREQUENCY))
            #import pdb ; pdb.set_trace()
            
            accuracy = 100 * correct / total
            print("Accuracy = {}".format(accuracy))
            if train_loss/OUTPUT_FREQUENCY < MIN_LOSS:
                torch.save(net.state_dict(), PATH)
           
            train_loss = 0.0
            
    
    print('Finished Training')
    
    net.eval()
    correct = 0
    total = 0
    for j, sample2 in enumerate(validation_loader, 0):
        
        input_image = sample2["image"].float()
        label = sample2["label"]
        
        output_class = net(input_image)
        #preds = torch.argmax(output_class, dim=1)
        print('validation',output_class)
        loss= criterion(output_class,label)
        valid_loss += loss.item()
        total += 1
        if label-6 < torch.round(output_class) and torch.round(output_class) < label+6:
            correct += 1
        print('correct',correct)
        if j % OUTPUT_FREQUENCY == OUTPUT_FREQUENCY - 1:    
            """ print every OUTPUT_FREQUENCY mini-batches"""
            plt.clf()
            print('[%d, %5d] validation loss: %.3f' %
                  (epoch + 1, j + 1, valid_loss /  OUTPUT_FREQUENCY))
            accuracy = 100 * correct / total
            print("Accuracy = {}".format(accuracy))
            valid_loss = 0.0