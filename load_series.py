# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:51:31 2021

@author: eser
"""
import SimpleITK as sitk
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import dicom_lesen as dl
import resize_image as res
table = pd.read_excel('much_better_table_with_classes.xlsx', sheet_name= 'with_classes')
directories = table['directory']
classes = table['Class']

for i in range(len(directories)):
    array = res.resize_images(directories[i],256)
    if array.any():
        np.save('images_%s.npy' % i,array)
        np.save('labels_%s.npy'% i, classes[i])
        
        print(i)
        
#print(cropped_images) 
