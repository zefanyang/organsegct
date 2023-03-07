#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/6/2021 4:25 PM
# @Author: yzf
import cv2
import matplotlib.pyplot as plt
from visualizers.normalize_orientation import *

def reverse_axes(data):
    return np.transpose(data, tuple(range(data.ndim))[::-1])

def read_image(image_file):
    image = sitk.ReadImage(image_file)
    data = sitk.GetArrayFromImage(image)
    data = reverse_axes(data)  # switch from zxy to xyz
    header = {
        'spacing': image.GetSpacing(),
        'origin': image.GetOrigin(),
        'direction': image.GetDirection()
    }
    return data, header

def save_image(data, header, output_file):
    data = reverse_axes(data)  # reverse back
    img_itk = sitk.GetImageFromArray(data)
    img_itk.SetSpacing(header['spacing'])
    img_itk.SetOrigin(header['origin'])
    if not isinstance(header['direction'], tuple):
        img_itk.SetDirection(header['direction'].flatten())
    else:
        img_itk.SetDirection(header['direction'])
    sitk.WriteImage(img_itk, output_file)

def show_image(data):
    center = (np.array(data.shape) - 1) // 2
    plt.subplot(131); plt.imshow(data[:, :, center[2]], cmap='gray')  # axial
    plt.subplot(132); plt.imshow(data[center[0], :, :], cmap='gray')  # sagittal
    plt.subplot(133); plt.imshow(data[:, center[1], :], cmap='gray')  # coronal

    plt.show()

file_src1 = './demo/raw_img/tcia_img0003.nii.gz'
file_src2 = './demo/raw_img/btcv_img0001.nii.gz'

data1, header1 = read_image(file_src1)
data1 = data1[:, :, ::-1]
header1['direction'] = np.eye(3)  # identity mapping
save_image(data1, header1, output_file='./demo/raw_img/tcia_img0003_new.nii.gz')
show_image(data1)

data2, header2 = read_image(file_src2)
header2['direction'] = np.eye(3)
save_image(data2, header2, output_file='./demo/raw_img/btcv_img0001_new.nii.gz')
show_image(data2)



# data1 = data1[:, :, ::-1]
# header1['direction'] = np.eye(3)


# save_image(data1, header1, './demo/tcia_img0003_new.nii.gz')

# data1, header1 = read_image(file_src1)
# # data2, header2 = read_image(file_src2)
#
# header1['original'] = header1.copy()
# # reorientation: transpose and flip
# cosine = np.asarray(header1['direction']).reshape(3, 3)
# inv_cosine = np.round(np.linalg.inv(cosine))
#
# transpose = np.argmax(abs(inv_cosine), axis=0)
# flip = np.sum(inv_cosine, axis=0)
# data1 = np.transpose(data1, tuple(transpose))
# data1 = data1[::int(flip[0]), ::int(flip[1]), ::int(flip[2])]
#
# header1['direction'] = np.eye(3)
#
# save_image(data1, header1, output_file='./demo/tcia_img0003_new.nii.gz')

# header2['original'] = header2.copy()
# # reorientation: transpose and flip
# cosine = np.asarray(header2['direction']).reshape(3, 3)
# inv_cosine = np.round(np.linalg.inv(cosine))
#
# transpose = np.argmax(inv_cosine, axis=0)
# flip = np.sum(inv_cosine, axis=0)
# data2 = np.transpose(data2, tuple(transpose))
# data2 = data2[::int(flip[0]), ::int(flip[1]), ::int(flip[2])]
#
# header2['direction'] = np.eye(3)
#
# save_image(data2, header2, output_file='./demo/btcv_img0001_new.nii.gz')





