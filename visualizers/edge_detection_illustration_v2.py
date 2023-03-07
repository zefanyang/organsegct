#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/17/2021 3:25 PM
# @Author: yzf
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
    'file': '../illustration/unet_edge_fold0/tcia_0005.npz',
    'index': 47
}

sample2 = {
    'file': '../illustration/unet_edge_fold0/tcia_0020.npz',
    'index': 42
}

sample3 = {
    'file': '../illustration/unet_edge_fold0/btcv_0039.npz',
    'index': 47
}

sample4 = {
    'file': '../illustration/unet_edge_fold2/btcv_0061.npz',
    'index': 45
}

ls = [sample1, sample2, sample3, sample4]

n = 4
sample_dct = sample4

data = np.load(sample_dct['file'])
spacing = data['spacing']
img = data['image'][..., sample_dct['index']]
seg = data['seg'][..., sample_dct['index']]
seg_map = data['seg_pr'][..., sample_dct['index']]
# edge_pr = data['edge_pr'][..., sample_dct['index']]

# resize image
new_size = get_new_size(img.shape, spacing)

# render image
img = (norm_score(img) * 255.).astype(np.uint8)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

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

# resize to the aspect ratio of spatial scale (mm)
seg_rgb = cv2.resize(seg_rgb, new_size, cv2.INTER_LINEAR)

# save .jpg image
cv2.imwrite(f'../illustration/edge_sample{n}.jpg', seg_rgb)




# # draw on plain canvas
# canvas = np.zeros(seg_rgb.shape, dtype=np.uint8)
# canvas = cv2.applyColorMap(canvas, cv2.COLORMAP_JET)
# cv2.drawContours(canvas, contours_ls, -1, (0, 0, 255), 2)
# canvas = cv2.addWeighted(canvas, 0.8, img_rgb, 0.2, 0)
# # draw on CT
# img_rgb = cv2.drawContours(img_rgb, contours_ls, -1, (0, 0, 255), 1)
