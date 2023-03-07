#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/30/2021 5:58 PM
# @Author: yzf
import os
import json
import SimpleITK as sitk

# fold = 3
# cv_json = '/raid/yzf/data/abdominal_ct/cv_high_resolution.json'
# out_fd = './volume_align'
#
# os.makedirs(out_fd+'/img', exist_ok=True)
# os.makedirs(out_fd+'/lab', exist_ok=True)
#
# with open(cv_json, 'r') as f:
#     data_dct = json.load(f)
#
# data = data_dct['val'][f'fold_{fold}']
# for file in data:
#     img_file = file[0]
#     lab_file = file[1]
#     simple_idx = '_'.join([img_file.split('/')[-3][:4], img_file.split('/')[-1][3:7]])
#
#     img = sitk.ReadImage(img_file)
#     lab = sitk.ReadImage(lab_file)
#     spacing = img.GetSpacing()
#
#     img_arr = sitk.GetArrayFromImage(img)  # z, x, y
#     lab_arr = sitk.GetArrayFromImage(lab)
#
#     if 'tcia' in simple_idx:
#         img_arr = img_arr[::-1, :, :]
#         lab_arr = lab_arr[::-1, :, :]
#
#     img_sitk = sitk.GetImageFromArray(img_arr)
#     lab_sitk = sitk.GetImageFromArray(lab_arr)
#
#     img_sitk.SetSpacing(spacing)
#     lab_sitk.SetSpacing(spacing)
#
#     sitk.WriteImage(img_sitk, out_fd + f'/img/{simple_idx}.nii.gz')
#     sitk.WriteImage(lab_sitk, out_fd + f'/lab/{simple_idx}.nii.gz')

networks = ['attentionunet', 'cascadedvnet', 'densevnet', 'unet++']
network = networks[3]
segmentations = f'./qualitative_comparison/segmentation/{network}/tcia_0042.nii.gz'
img = sitk.ReadImage(segmentations)
spacing = img.GetSpacing()
img_arr = sitk.GetArrayFromImage(img)
img_sitk = sitk.GetImageFromArray(img_arr)
img_sitk.SetSpacing(spacing)
sitk.WriteImage(img_sitk, f'./qualitative_comparison/segmentation/{network}/tcia_0042_new.nii.gz')
