#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 3/8/2023 6:52 PM
# @Author: yzf
"""Detect edge maps"""
import os
import json
import numpy as np
from skimage import measure
import SimpleITK as sitk

basepath = '/data/yzf/dataset/organct/external/preprocessed'
metadata = '/data/yzf/dataset/organct/external/dataset.json'
f = open(metadata)
metadata = json.load(f)
f.close()
trainings = metadata['training']

def label_to_edge(label, zaxis):
    shape = label.shape
    edge_map = np.zeros_like(label)
    for z in range(shape[zaxis]):
        slice_ = label[z, :, :]
        boundaries = measure.find_contours(slice_, level=0.1)
        for boundary in boundaries:
            boundary = boundary.astype(int)
            for k in range(len(boundary)):
                i = boundary[k, 0]
                j = boundary[k, 1]
                edge_map[z, i, j] = 1
    return edge_map

if __name__ == '__main__':
    for i in range(len(trainings)):
        name = os.path.basename(trainings[i]['label'])
        # preprocessed label
        labpath = os.path.join(basepath, 'label', name)
        lab = sitk.ReadImage(labpath)
        o, s, d = lab.GetOrigin(), lab.GetSpacing(), lab.GetDirection()
        labarr = sitk.GetArrayFromImage(lab)  # (z, x, y)
        edgearr = label_to_edge(labarr, zaxis=0)
        edge = sitk.GetImageFromArray(edgearr)
        edge.SetOrigin(o); edge.SetSpacing(s); edge.SetDirection(d)
        sitk.WriteImage(edge, os.path.join(basepath, 'edge', name))
        print(name)


