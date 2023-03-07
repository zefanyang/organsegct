#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/18/2021 3:49 PM
# @Author: yzf
"""Example of finding contours in OpenCV"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

im = np.zeros((100, 100)).astype(np.uint8)
im[20: -20, 20: -20] = 255.
cnts, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
im2 = cv2.drawContours(im, cnts, 0, (0, 255, 0), 1)
cv2.imwrite('./test.jpg', im2)
# plt.imshow(im2)
# plt.axis('off')
# plt.show()
