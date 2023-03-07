#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/17/2021 3:25 PM
# @Author: yzf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from visualizers.batch_visualizer import *

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
    'index': 44
}

ls = [sample1, sample2, sample3, sample4]

n = 3  # TODO
sample_dct = sample3  # TODO

data = np.load(sample_dct['file'])
spacing = data['spacing']
img = data['image'][..., sample_dct['index']]
edge = data['edge'][..., sample_dct['index']]
edge_pr = data['edge_pr'][..., sample_dct['index']]

# resize image
h, w = img.shape  # pixels
height, width = h * spacing[0], w * spacing[1]
ratio = width / height

new_h = int(h * ratio)
new_size = (new_h, w)

# render image
img = (norm_score(img) * 255.).astype(np.uint8)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

edge_pr = (norm_score(edge_pr) * 255.).astype(np.uint8)
edge_pr_rgb = cv2.applyColorMap(edge_pr, cv2.COLORMAP_JET)
edge_pr_rgb = cv2.addWeighted(edge_pr_rgb, 0.8, img_rgb, 0.2, 0)

# img = cv2.resize(img, new_size, cv2.INTER_LINEAR)
# edge = cv2.resize(edge, new_size, cv2.INTER_NEAREST)
# edge_pr_rgb = cv2.resize(edge_pr_rgb, new_size, cv2.INTER_LINEAR)

edge = np.where(edge > 0., 255, 0)
edge = edge.astype(np.uint8)
# edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
# edge[..., 0] = 0
# edge[..., 2] = 0

seg = data['seg'][..., sample_dct['index']]
seg_rgb = get_score_map(seg, rang=(0, 8))
# seg_contour = cv2.addWeighted(seg_rgb, 1, edge, 1, 0)
#
# cv2.imwrite('./test.jpg', seg_contour)

# seg_rgb = get_score_map(seg, rang=(0, 8))
seg_rgb = (norm_score(seg) * 255.).astype(np.uint8)
# seg_rgb = cv2.resize(seg_rgb, new_size, cv2.INTER_LINEAR)
x_ls, y_ls = np.where(edge > 0)
cnts = [np.asarray([[y, x] for y, x in zip(y_ls, x_ls)])[:, None, :]]
cv2.drawContours(seg_rgb, cnts, 0, (255, 0, 0), 1)

cv2.imwrite('./test.jpg', seg_rgb)

# cv2.imwrite(f'../illustration/sample{n}_img.jpg', img)
# cv2.imwrite(f'../illustration/sample{n}_edge.jpg', edge)
# cv2.imwrite(f'../illustration/sample{n}_edge_pr.jpg', edge_pr_rgb)