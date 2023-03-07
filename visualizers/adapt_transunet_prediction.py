#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 8/9/2021 9:51 PM
# @Author: yzf
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import os
ind = 5
cases = ['btcv_0001.nii.gz', 'btcv_0010.nii.gz', 'btcv_0037.nii.gz', 'tcia_0042.nii.gz', 'btcv_0027.nii.gz', 'btcv_0077.nii.gz']
img_fd = './qualitative_comparison/transunet/img/'
pred_fd = './qualitative_comparison/transunet/transunet/'

new_pred_fd = './qualitative_comparison/transunet/new_transunet/'
os.makedirs(new_pred_fd, exist_ok=True)

spacing = sitk.ReadImage(img_fd + cases[ind]).GetSpacing()
pred_arr = nib.load(pred_fd + cases[ind]).get_fdata()
pred_sitk = sitk.GetImageFromArray(pred_arr)
pred_sitk.SetSpacing(spacing)
sitk.WriteImage(pred_sitk, new_pred_fd+cases[ind])
