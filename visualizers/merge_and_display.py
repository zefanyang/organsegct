#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 6/30/2021 4:07 PM
# @Author: yzf
"""Merge .jpg image and save for comparison"""
import cv2
from pathlib import Path
import glob
from visualizers.image_tools import *

proposed_fd = Path('./qualitative_comparison/snapshots_for_merging/proposed')
unet_fd = Path('./qualitative_comparison/snapshots_for_merging/unet')
vnet_fd = Path('./qualitative_comparison/snapshots_for_merging/vnet')

# output folder
out_fd = Path('./qualitative_comparison/snapshots_merged')
out_fd.mkdir(exist_ok=True)

proposed_ls = sorted(proposed_fd.glob('*.jpg'))
unet_ls = sorted(unet_fd.glob('*.jpg'))
vnet_ls = sorted(vnet_fd.glob('*.jpg'))

width = 489 // 3
clip = lambda image: image[:, -width:, ]

for im1, im2, im3 in zip(proposed_ls, unet_ls, vnet_ls):
    image1 = cv2.imread(str(im1))

    image2 = cv2.imread(str(im2))
    image3 = cv2.imread(str(im3))

    # Clip
    image2 = clip(image2)
    image3 = clip(image3)

    h_image = imhstack([image1, image2, image3])

    imwrite(str(out_fd / im1.name), h_image)


# # baseline
# task_fd1 = Path('../output/unet_edge/snapshots')
# # comparison
# task_fd2 = Path('../output/unet_edge_skip/snapshots')
# # output folder
# out_fd = Path('./baseline_vs_unet_edge_skip')
# out_fd.mkdir(exist_ok=False)
#
# im_ls1 = sorted(task_fd1.glob('*.jpg'))
# im_ls2 = sorted(task_fd2.glob('*.jpg'))
#
# for im1, im2 in zip(im_ls1, im_ls2):
#     image1 = cv2.imread(str(im1))
#     image2 = cv2.imread(str(im2))
#
#     h_image = imhstack([image1, image2])
#
#     imwrite(str(out_fd / im1.name), h_image)





