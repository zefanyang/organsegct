#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/22/2021 10:52 PM
# @Author: yzf
import os
import cv2
import torch
import numpy as np

sample4 = {
    'file': '../illustration/unet_edge_fold2/btcv_0061.npz',
    'index': 45
}

out_fd = '../illustration/intermediate_results'
os.makedirs(out_fd, exist_ok=True)

def norm_score(score, rang=None):
    """Min-max scaler"""
    min_ = score.min()
    max_ = score.max()
    if rang is not None:
        min_ = min(rang)
        max_ = max(rang)
    return (score - min_) / max(max_ - min_, 1e-5)

def get_seg_map(input):
    input = torch.from_numpy(input)
    prob = torch.softmax(input, dim=1)
    # map_ = torch.argmax(prob, dim=1, keepdim=True)
    map_ = prob[:, 5:6, ...]

    return map_.numpy()

def get_seg_color_map(input):
    input = (norm_score(input, rang=(0, 8)) * 255.).astype(np.uint8)
    input = cv2.applyColorMap(input, cv2.COLORMAP_JET)
    return input


data_dict = np.load('../illustration/btcv_0061_intermediate_results.npz')

# seg_score = data_dict['seg_score']
# rfp_seg_score = data_dict['rfp_seg_score']
# comb_seg_score = data_dict['comb_seg_score']
# edge_score = data_dict['edge_score']
# dsup4 = data_dict['dsup4']
# dsup3 = data_dict['dsup3']
# dsup2 = data_dict['dsup2']
# dsup1 = data_dict['dsup1']

obj_dict = {
    "1": "spleen",
    "2": 'left kidney',
    "3": 'gallbladder',
    "4": 'esophagus',
    "5": 'liver',
    "6": 'stomach',
    "7": 'pancreas',
    "8": 'duodenum',
}

def get_color_map(obj_prob, colormap=cv2.COLORMAP_JET):
    norm_prob = (norm_score(obj_prob) * 255.).astype(np.uint8)
    norm_prob_rgb = cv2.applyColorMap(norm_prob, colormap)
    return norm_prob_rgb

# ['seg_score', 'rfp_seg_score', 'comb_seg_score', 'edge_score' , 'dsup4', 'dsup3', 'dsup2', 'dsup1']
obj_score_name = data_dict.files[7]  # TODO 8 files in total
obj_score = data_dict[obj_score_name]

obj_score = torch.from_numpy(obj_score)
obj_score = torch.softmax(obj_score, dim=1)
obj_score = obj_score.numpy()
for idx in range(1, 9):
    obj_prob = obj_score[0, idx, :, :, sample4['index']]
    obj_prob_rgb = get_color_map(obj_prob)
    cv2.imwrite(out_fd+f'/{obj_score_name}_{obj_dict[str(idx)]}.jpg', obj_prob_rgb)

# dsup4_prob = dsup4_prob.numpy()
# dsup4_liver = dsup4_prob[0, 6, :, :, sample4['index']]  # hard-coding
# dsup4_liver_rgb = get_color_map(dsup4_liver)
# cv2.imwrite(out_fd+'/dsup4_liver.jpg', dsup4_liver_rgb)



# dsup4_liver = (norm_score(dsup4_liver) * 255.).astype(np.uint8)
# dsup4_liver_rgb = cv2.applyColorMap(dsup4_liver, cv2.COLORMAP_JET)

# cv2.imwrite(out_fd+'/dsup4_liver.jpg', dsup4_liver_rgb)




# map_ = get_seg_map(dsup4)
# map_rgb = get_seg_color_map(map_[0, 0, :, :, sample4['index']])

# dsup4 = torch.softmax(dsup4, dim=1)
# dsup4 = np.argmax(dsup4, axis=1)[0][..., sample4['index']]
# dsup4 = (norm_score(dsup4, rang=(0, 8)) * 255.).astype(np.uint8)
# dsup4_rgb = cv2.applyColorMap(dsup4, cv2.COLORMAP_JET)

# cv2.imwrite(out_fd + '/dsup4.jpg', map_rgb)
# cv2.imwrite(out_fd + '/dsup3.jpg', dsup3_rgb)
# cv2.imwrite(out_fd + '/dsup2.jpg', dsup2_rgb)
# cv2.imwrite(out_fd + '/dsup1.jpg', dsup1_rgb)

