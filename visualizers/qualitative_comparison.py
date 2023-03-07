#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/30/2021 10:06 PM
# @Author: yzf
"""Qualitative comparison of segmentation"""
import os
import nibabel as nib
import SimpleITK as sitk
import cv2
import numpy as np

def norm_score(score, rang=None):
    """Min-max scaler"""
    min_ = score.min()
    max_ = score.max()
    if rang is not None:
        min_ = min(rang)
        max_ = max(rang)
    return (score - min_) / max(max_ - min_, 1e-5)

def get_new_size(shape, spacing):
    h, w = shape  # pixels
    height, width = h * spacing[0], w * spacing[1]  # mm
    ratio = width / height

    new_h = int(h * ratio)
    new_size = (new_h, w)
    return new_size

# def clip_intensity(arr):
#     arr = np.clip(arr, -250., 200.)
#     arr = norm_score(arr)
#     return (arr * 255.).astype(np.uint8)
#
# def get_score_map(score, rang=None, TYPE=cv2.COLORMAP_JET):
#     score = norm_score(score, rang=rang)
#     score_cmap = cv2.applyColorMap((score * 255.99).astype(np.uint8), TYPE)
#     return score_cmap

# 13 samples
samples = [{'name': 'btcv_0075', 'slc': 37},
           {'name': 'btcv_0010', 'slc': 25},
           {'name': 'btcv_0031', 'slc': 34},
           {'name': 'btcv_0037', 'slc': 30},
           {'name': 'tcia_0047', 'slc': 29},
           {'name': 'btcv_0001', 'slc': 36},
           {'name': 'btcv_0026', 'slc': 31},
           {'name': 'btcv_0033', 'slc': 29},
           {'name': 'btcv_0061', 'slc': 36},
           {'name': 'btcv_0077', 'slc': 30},
           {'name': 'btcv_0063', 'slc': 38},
           {'name': 'tcia_0025', 'slc': 33},
           {'name': 'tcia_0042', 'slc': 38}]

# image folder
img_fd = './qualitative_comparison/volume_align/img'
# gt folder
gt_fd = './qualitative_comparison/volume_align/lab'
# proposed predictions
proposed_fd = './qualitative_comparison/predictions/proposed'
# unet predictions
unet_fd = './qualitative_comparison/predictions/unet'
# vnet predictions
vnet_fd = './qualitative_comparison/predictions/vnet'

# output folder
out_fd = './qualitative_comparison/representative_samples2'
os.makedirs(out_fd, exist_ok=True)

# 0-12
# idx = 3
for idx in range(13):
    samp = samples[idx]
    name = samp['name']
    slc = samp['slc']

    subj_out = out_fd+f'/{name}_{slc}'
    os.makedirs(subj_out, exist_ok=True)

    img = nib.load(img_fd+f'/{name}.nii.gz').get_fdata()[..., slc]
    gt = nib.load(gt_fd+f'/{name}.nii.gz').get_fdata()[..., slc]
    proposed = nib.load(proposed_fd+f'/{name}.nii.gz').get_fdata()[..., slc]
    unet = nib.load(unet_fd+f'/{name}.nii.gz').get_fdata()[..., slc]
    vnet = nib.load(vnet_fd+f'/{name}.nii.gz').get_fdata()[..., slc]

    spacing = sitk.ReadImage(img_fd+f'/{name}.nii.gz').GetSpacing()
    new_size = get_new_size(img.shape, spacing)

    img = np.clip(img, -250., 200)
    img = (norm_score(img) * 255.).astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(subj_out+'/img.jpg', img)

    gt = (norm_score(gt, rang=(0, 8)) * 255.).astype(np.uint8)
    gt = cv2.applyColorMap(gt, colormap=cv2.COLORMAP_JET)
    gt = cv2.addWeighted(gt, 0.8, img_rgb, 0.2, 0)
    # gt = cv2.resize(gt, new_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(subj_out+'/gt.jpg', gt)

    proposed = (norm_score(proposed, rang=(0, 8)) * 255.).astype(np.uint8)
    proposed = cv2.applyColorMap(proposed, colormap=cv2.COLORMAP_JET)
    proposed = cv2.addWeighted(proposed, 0.8, img_rgb, 0.2, 0)
    # proposed = cv2.resize(proposed, new_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(subj_out+'/proposed.jpg', proposed)

    unet = (norm_score(unet, rang=(0, 8)) * 255.).astype(np.uint8)
    unet = cv2.applyColorMap(unet, colormap=cv2.COLORMAP_JET)
    unet = cv2.addWeighted(unet, 0.8, img_rgb, 0.2, 0)
    # unet = cv2.resize(unet, new_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(subj_out+'/unet.jpg', unet)

    vnet = (norm_score(vnet, rang=(0, 8)) * 255.).astype(np.uint8)
    vnet = cv2.applyColorMap(vnet, colormap=cv2.COLORMAP_JET)
    vnet = cv2.addWeighted(vnet, 0.8, img_rgb, 0.2, 0)
    # vnet = cv2.resize(vnet, new_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(subj_out+'/vnet.jpg', vnet)




