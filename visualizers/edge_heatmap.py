#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 3/15/2021 4:36 PM
# @Author: yzf
import numpy as np
import nibabel as nib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

demo_file = './btcv_high_resolution_edge0009.nii.gz'

image = nib.load(demo_file).get_data()

percent = .5
ind = 42

max_ = percent * image.max()
image_ = image.copy()
image_[image < max_] = 0.

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(image[..., ind], cmap=cm.jet)
ax2.imshow(image_[..., ind], cmap=cm.jet)
plt.show()

