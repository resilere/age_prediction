# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:55:19 2021

@author: eser
"""
import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
table = pd.read_excel('much_better_table_with_ages.xlsx', sheet_name= 'with_ages')
directories = table['directory']

directory = directories[0]

def downsamplePatient(patient_CT, resize_factor):

    original_CT = patient_CT
    dimension = original_CT.GetDimension()
    
    
            
    reference_physical_size = np.zeros(original_CT.GetDimension())
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size)]
    
    reference_origin = original_CT.GetOrigin()
    reference_direction = original_CT.GetDirection()
    print(original_CT.GetSize())
    reference_size = [original_CT.GetSize()[0],original_CT.GetSize()[1],round(original_CT.GetSize()[2]/resize_factor)] 
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

    reference_image = sitk.Image(reference_size, original_CT.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)
    
    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(original_CT.GetDirection())

    transform.SetTranslation(np.array(original_CT.GetOrigin()) - reference_origin)
     
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(original_CT.TransformContinuousIndexToPhysicalPoint(np.array(original_CT.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform=sitk.CompositeTransform([centered_transform, centering_transform])
    #sitk.Show(sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0))
    
    return sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0)
def resize_images(directory,target_size):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
        
    image = reader.Execute()
    size = image.GetSize()[2]
    resize_factor = size/target_size
    dimension = image.GetDimension()
    print('dimension',dimension)
    arr2 = np.array([])
    if dimension == 3 :
        image2= downsamplePatient(image, resize_factor)
    
        print(image2.GetSize())
        arr2= sitk.GetArrayFromImage(image2)
        #arr = sitk.GetArrayFromImage(image)
    
# =============================================================================
#     fig, axes = plt.subplots(nrows = 1, ncols = 2)
#     axes[0].imshow(arr[350,:,:])
#     axes[1].imshow(arr2[230,:,:])
# =============================================================================
    return arr2
resize_images(directory,256)
