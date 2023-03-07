#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/17/2021 3:25 PM
# @Author: yzf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def norm_score(score, rang=None):
    """Min-max scaler"""
    min_ = score.min()
    max_ = score.max()
    if rang is not None:
        min_ = min(rang)
        max_ = max(rang)
    return (score - min_) / max(max_ - min_, 1e-5)

def get_score_map(score, rang=None, TYPE=cv2.COLORMAP_JET):
    score = norm_score(score, rang=rang)
    score_cmap = cv2.applyColorMap((score * 255.99).astype(np.uint8), TYPE)
    return score_cmap

def get_new_size(shape, spacing):
    h, w = shape  # pixels
    height, width = h * spacing[0], w * spacing[1]  # mm
    ratio = width / height

    new_h = int(h * ratio)
    new_size = (new_h, w)
    return new_size


# four samples
sample1 = {
    'file': '../illustration/BIBM/edge_data/tcia_0005.npz',
    'index': 35
}

sample2 = {
    'file': '../illustration/BIBM/edge_data/tcia_0008.npz',
    'index': 23
}

sample3 = {
    'file': '../illustration/BIBM/edge_data/tcia_0016.npz',
    'index': 23
}

out_fd = f'../illustration/BIBM/outputs/'
os.makedirs(out_fd, exist_ok=True)

ls = [sample1, sample2, sample3]
n = 3
sample_dct = ls[n]

data = np.load(sample_dct['file'])
spacing = data['spacing']
img = data['img'][..., sample_dct['index']]
seg = data['label'][..., sample_dct['index']]
edge_pr = data['edge'][..., sample_dct['index']]
# edge_pr = data['edge_pr'][..., sample_dct['index']]

# render image
img = np.clip(img, -250, 200)
img = (norm_score(img) * 255.).astype(np.uint8)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# resize image
new_size = get_new_size(img.shape, spacing)

# =================================Edge Ground Truth=================================

# # seg map
# seg_map_rgb = (norm_score(seg_map, rang=(0, 8)) * 255.).astype(np.uint8)
# seg_map_rgb = cv2.applyColorMap(seg_map_rgb, cv2.COLORMAP_JET)
#
# seg_map_rgb = cv2.addWeighted(seg_map_rgb, 0.8, img_rgb, 0.2, 0)
# seg_map_rgb = cv2.resize(seg_map_rgb, new_size, cv2.INTER_LINEAR)
# cv2.imwrite(f'../illustration/seg_map{n}.jpg', seg_map_rgb)

seg_rgb = (norm_score(seg, rang=(0, 8)) * 255.).astype(np.uint8)
seg_rgb = cv2.applyColorMap(seg_rgb, cv2.COLORMAP_JET)

# find contour of each object
label_set = np.unique(seg)
obj_maps = [(np.where(seg==l, 1, 0) * 255).astype(np.uint8)
            for l in label_set[1:]]
contours_ls = []
for obj in obj_maps:
    contours, _ = cv2.findContours(obj, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours_ls.append(contours[0])

seg_rgb = cv2.drawContours(image=seg_rgb, contours=contours_ls, contourIdx=-1, color=(0, 0, 255), thickness=2)
seg_rgb = cv2.addWeighted(seg_rgb, 0.8, img_rgb, 0.2, 0)

# resize to the aspect ratio of spatial scale (mm) and save .jpg image
seg_rgb = cv2.resize(seg_rgb, new_size, cv2.INTER_LINEAR)
cv2.imwrite(f'{out_fd}/edge_ground_truth{n}.jpg', seg_rgb)
# =================================Edge Ground Truth=================================
# edge probability
edge_pr = (norm_score(edge_pr) * 255.).astype(np.uint8)
edge_pr_rgb = cv2.applyColorMap(edge_pr, cv2.COLORMAP_JET)
edge_pr_rgb = cv2.addWeighted(edge_pr_rgb, 0.8, img_rgb, 0.2, 0)

img = cv2.resize(img, new_size, cv2.INTER_LINEAR)
edge_pr_rgb = cv2.resize(edge_pr_rgb, new_size, cv2.INTER_LINEAR)
cv2.imwrite(f'{out_fd}/img{n}.jpg', img)
cv2.imwrite(f'{out_fd}/edge_pr_rgb{n}.jpg', edge_pr_rgb)
# ===================================================================================


# # draw on plain canvas
# canvas = np.zeros(seg_rgb.shape, dtype=np.uint8)
# canvas = cv2.applyColorMap(canvas, cv2.COLORMAP_JET)
# cv2.drawContours(canvas, contours_ls, -1, (0, 0, 255), 2)
# canvas = cv2.addWeighted(canvas, 0.8, img_rgb, 0.2, 0)
# # draw on CT
# img_rgb = cv2.drawContours(img_rgb, contours_ls, -1, (0, 0, 255), 1)


# # save data as .npz
# ind = 0
# cases = ['tcia_0005', 'tcia_0008', 'tcia_0016']
#
# img_fd = '../illustration/BIBM/edge_data/image'
# edge_fd = '../illustration/BIBM/edge_data/edge'
# label_fd = '../illustration/BIBM/edge_data/label'
#
# import nibabel as nib
# import SimpleITK as sitk
#
# # img = nib.load(f'{img_fd}/{cases[ind]}.nii.gz').get_fdata()
# # edge = nib.load(f'{edge_fd}/{cases[ind]}.nii.gz').get_fdata()
# # label = nib.load(f'{label_fd}/{cases[ind]}.nii.gz').get_fdata()
#
# img = nib.load(f'{img_fd}/{cases[ind]}.nii.gz').get_fdata()[:, :, ::-1]
# edge = nib.load(f'{edge_fd}/{cases[ind]}.nii.gz').get_fdata()
# label = nib.load(f'{label_fd}/{cases[ind]}.nii.gz').get_fdata()[:, :, ::-1]
#
#
# spacing = sitk.ReadImage(f'{img_fd}/{cases[ind]}.nii.gz').GetSpacing()
#
# np.savez(file=f'../illustration/BIBM/edge_data/{cases[ind]}', img=img, label=label, edge=edge, spacing=spacing)
