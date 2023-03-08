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

        name = os.path.basename(imgpath)
        sitk.WriteImage(imgnew, os.path.join(outpath, 'image', name))
        sitk.WriteImage(labnew, os.path.join(outpath, 'label', name))
        print(name)

