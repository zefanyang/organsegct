#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 6/24/2021 11:29 AM
# @Author: yzf
"""Visualize segmentation results for performance comparison"""
import nibabel as nib
import cv2
from visualizers.image_tools import *

def get_nii_data(file):
    img = nib.load(file)
    arr = img.get_fdata()  # h, w, d
    return arr

def norm_score(score, rang=None):
    """Min-max scaler"""
    min_ = score.min()
    max_ = score.max()
    if rang is not None:
        min_ = min(rang)
        max_ = max(rang)
    return (score - min_) / max(max_ - min_, 1e-5)

def clip_intensity(arr):
    arr = np.clip(arr, -250., 200.)
    arr = norm_score(arr)
    return (arr * 255.).astype(np.uint8)

def get_score_map(score, rang=None, TYPE=cv2.COLORMAP_JET):
    score = norm_score(score, rang=rang)
    score_cmap = cv2.applyColorMap((score * 255.99).astype(np.uint8), TYPE)
    return score_cmap

# def get_score_map(score, TYPE=cv2.COLORMAP_JET):
#     score = norm_score(score)
#     score_cmap = cv2.applyColorMap((score * 255.99).astype(np.uint8), TYPE)
#     return score_cmap

# img = './demo/img0001.nii.gz'
# seg = './demo/btcv_high_resolution_pseg0001.nii.gz'
# edge = './demo/btcv_high_resolution_edge0001.nii.gz'

# img = './demo/img0005.nii.gz'
# seg = './demo/btcv_high_resolution_pseg0005.nii.gz'
# edge = './demo/btcv_high_resolution_edge0005.nii.gz'
#
# img = get_nii_data(img)
# seg = get_nii_data(seg)
# edge = get_nii_data(edge)
# assert img.shape == seg.shape == edge.shape
#
# h, w, d = img.shape
# v_images = []

# for ind in range(d):
#     im = np.rot90(img[..., ind])
#     se = np.rot90(seg[..., ind])
#     ed = np.rot90(edge[..., ind])
#
#     im = clip_intensity(im)
#     imRGB = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB);
#     getScoreMap = lambda x: cv2.addWeighted(get_score_map(x),
#                                             0.8, imRGB, 0.2, 0)
#     se = getScoreMap(se)
#     ed = getScoreMap(ed)
#
#     im = imtext(im, text='{:0d}'.format(ind), space=(3, 14), color=(255,)*3, thickness=1, fontScale=.6)
#
#
#     h_images = [im, se, ed]
#     v_images.append(imhstack(h_images, height=120))
# v_images = imvstack(v_images)
# imwrite(os.path.join('./demo', 'abdominal_ct.jpg'), v_images)





# # cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) → dst
# # src1: get_score_map(cv2.resize(x, (args.image_size,) * 2)[:h, :w]),
# # alpha: 0.8,
# # src2: image,
# # beta: 0.2,
# # gamma: 0
# # dst = src1 * alpha + src2 * beta + gamma

# # cv2.resize(x, (args.image_size,) * 2)[:h, :w]
# # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst
# # src: x
# # dsize: (321, 321, )
# getScoreMap = lambda x: cv2.addWeighted(get_score_map(cv2.resize(x, (args.image_size,) * 2)[:h, :w]),
#                                         0.8, image, 0.2, 0)
# #
# # x = -1 return [0, 1]; x = 1 return [1, 0]
# visScores = lambda x: [getScoreMap(np.maximum(x, 0)), getScoreMap(np.maximum(-x, 0))]

# class Args:
#     def __init__(self):
#         pass
# args = Args()
# args.num_cls = 20
# v_images = [[] for _ in range(args.num_cls)]  # empty list for each class
# NUM = 20 # number of rows
#
# for n_batch, batch in enumerate(loader, 1):
#     mod.forward(batch, is_train=False)
#     outputs = [x.asnumpy() for x in mod.get_outputs()]
#     cam, icd_bu, icd_bu_sp, icd_td = outputs
#
#     image_src_list = loader.cache_image_src_list
#     label = batch.label[0].asnumpy()
#     N, C, H, W = icd_td.shape
#     for img_src, label, cam, icd_bu, icd_bu_sp, icd_td in zip(image_src_list, label, cam, icd_bu, icd_bu_sp, icd_td):
#         Ls = np.nonzero(label)[0]  # number of classes
#         cam = cam[Ls]
#         icd = icd_td[Ls]
#
#         name = os.path.basename(img_src).rsplit('.', 1)[0]
#         npsave(os.path.join(args.snapshot, 'results', 'scores_cam', name + '.npy'), cam)
#         npsave(os.path.join(args.snapshot, 'results', 'scores_icd', name + '.npy'), icd)
#
#         # demos
#         # image contains only one class
#         if len(Ls) == 1:
#             L = Ls[0]  # class L
#             # if smaller than NUM (20) rows
#             if len(v_images[L]) < NUM:
#                 image = cv2.imread(img_src)
#                 h, w = image.shape[:2]
#
#                 # add score map upon raw image
#                 getScoreMap = lambda x: cv2.addWeighted(get_score_map(cv2.resize(x, (args.image_size,) * 2)[:h, :w]),
#                                                         0.8, image, 0.2, 0)
#                 # x = -1 return [0, 1]; x = 1 return [1, 0]
#                 # foreground and background
#                 visScores = lambda x: [getScoreMap(np.maximum(x, 0)), getScoreMap(np.maximum(-x, 0))]
#
#                 # sum([[1], [2]], []) return [1, 2]
#                 h_images = sum([[image]] + list(map(visScores, [cam[0], icd_bu[L], icd_bu_sp[L], icd_td[0]])), [])
#                 # store horizontally stacked images for each class
#                 v_images[L].append(imhstack(h_images, height=120))
#             # if get enough images for class L
#             elif len(v_images[L]) == NUM:
#                 # vertically stack image
#                 img = imvstack(v_images[L])
#                 imwrite(os.path.join(args.snapshot, 'results', 'scores_demo', 'class_%d.jpg' % L), img)
#                 v_images[L].append(None)
#
# # in case that the number of samples for some classes are smaller than NUM (20).
# for L, v_images in enumerate(v_images):
#     if v_images and v_images[-1] is not None:
#         img = imvstack(v_images)
#         imwrite(os.path.join(args.snapshot, 'results', 'scores_demo', 'class_%d.jpg' % L), img)