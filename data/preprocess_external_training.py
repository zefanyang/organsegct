#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 3/8/2023 4:39 PM
# @Author: yzf
"""Resample the volume"""

import os
import json
import numpy as np
import SimpleITK as sitk

basepath = '/data/yzf/dataset/organct/external'
outpath = '/data/yzf/dataset/organct/external/preprocessed'
metadata = '/data/yzf/dataset/organct/external/dataset.json'
f = open(metadata)
metadata = json.load(f)
f.close()
trainings = metadata['training']
resample_size = (160, 160, 64)  # (x, y, z)

wordlabels = {
    "background": "0",
    "liver": "1",
    "spleen": "2",
    "left_kidney": "3",
    "right_kidney": "4",
    "stomach": "5",
    "gallbladder": "6",
    "esophagus": "7",
    "pancreas": "8",
    "duodenum": "9",
    "colon": "10",
    "intestine": "11",
    "adrenal": "12",
    "rectum": "13",
    "bladder": "14",
    "Head_of_femur_L": "15",
    "Head_of_femur_R": "16"
}

dvnetlabels = {
    "spleen": "1",
    'left_kidney': "2",
    'gallbladder': "3",
    'esophagus': "4",
    'liver': "5",
    'stomach': "6",
    'pancreas': "7",
    'duodenum': "8",
}

def class_mapping(label):
    labelnew = np.zeros_like(label)
    labelnew[label == int(wordlabels['spleen'])] = int(dvnetlabels['spleen'])
    labelnew[label == int(wordlabels['left_kidney'])] = int(dvnetlabels['left_kidney'])
    labelnew[label == int(wordlabels['gallbladder'])] = int(dvnetlabels['gallbladder'])
    labelnew[label == int(wordlabels['esophagus'])] = int(dvnetlabels['esophagus'])
    labelnew[label == int(wordlabels['liver'])] = int(dvnetlabels['liver'])
    labelnew[label == int(wordlabels['stomach'])] = int(dvnetlabels['stomach'])
    labelnew[label == int(wordlabels['pancreas'])] = int(dvnetlabels['pancreas'])
    labelnew[label == int(wordlabels['duodenum'])] = int(dvnetlabels['duodenum'])
    return labelnew

def resample_to_spacing(image, new_spacing, is_mask=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(original_size[0]*(original_spacing[0]/new_spacing[0]))),
                int(round(original_size[1]*(original_spacing[1]/new_spacing[1]))),
                int(round(original_size[2]*(original_spacing[2]/new_spacing[2])))]
    if not is_mask:
        resampled_img = sitk.Resample(image, new_size, sitk.Transform(),
                                      sitk.sitkLinear, image.GetOrigin(),
                                      new_spacing, image.GetDirection(), 0.0,
                                      image.GetPixelID())
    else:
        resampled_img = sitk.Resample(image, new_size, sitk.Transform(),
                                      sitk.siktNearestNeighbor, image.GetOrigin(),
                                      new_spacing, image.GetDirection(), 0.0,
                                      image.GetPixelID())
    return resampled_img

def resample_to_size(img, new_size, is_mask=False):
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_spacing = (float(original_size[0] * original_spacing[0]) / new_size[0],
                   float(original_size[1] * original_spacing[1]) / new_size[1],
                   float(original_size[2] * original_spacing[2]) / new_size[2])

    if not is_mask:
        resampled_img = sitk.Resample(img, new_size, sitk.Transform(),
                                      sitk.sitkLinear, img.GetOrigin(),
                                      new_spacing, img.GetDirection(), 0.0,
                                      img.GetPixelID())
    else:
        resampled_img = sitk.Resample(img, new_size, sitk.Transform(),
                                      sitk.sitkNearestNeighbor, img.GetOrigin(),
                                      new_spacing, img.GetDirection(), 0.0,
                                      img.GetPixelID())
    return resampled_img

if __name__ == '__main__':
    # Resample images
    for i in range(len(trainings)):
        imgpath = os.path.join(basepath, trainings[i]['image'][2:])
        labpath = os.path.join(basepath, trainings[i]['label'][2:])
        img = sitk.ReadImage(imgpath)
        lab = sitk.ReadImage(labpath)

        # Sanity check
        assert img.GetSpacing() == lab.GetSpacing() and img.GetSize() == lab.GetSize() and \
        img.GetOrigin() == lab.GetOrigin() and img.GetDirection() == lab.GetDirection()

        imgnew = resample_to_size(img, new_size=resample_size, is_mask=False)
        labnew = resample_to_size(lab, new_size=resample_size, is_mask=True)
        onew, dnew, snew = labnew.GetOrigin(), labnew.GetDirection(), labnew.GetSpacing()

        # Map classes
        labarr = sitk.GetArrayFromImage(labnew)
        labarrmapped = class_mapping(labarr)
        labarrmappedsitk = sitk.GetImageFromArray(labarrmapped)
        labarrmappedsitk.SetOrigin(onew); labarrmappedsitk.SetDirection(dnew); labarrmappedsitk.SetSpacing(snew)

        name = os.path.basename(imgpath)
        sitk.WriteImage(imgnew, os.path.join(outpath, 'image', name))
        sitk.WriteImage(labarrmappedsitk, os.path.join(outpath, 'label', name))
        print(name)
