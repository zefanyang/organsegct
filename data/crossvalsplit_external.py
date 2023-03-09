#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 3/8/2023 8:10 PM
# @Author: yzf
"""Split data into 4 folds"""
import os
import json
import numpy as np

basepath = '/data/yzf/dataset/organct/external'
metadata = '/data/yzf/dataset/organct/external/dataset.json'
f = open(metadata)
metadata = json.load(f)
f.close()
trainings = metadata['training']

def make_cvjson():
    names = np.array([os.path.basename(_['image']) for _ in trainings])
    np.random.seed(123)
    indices = np.arange(0, 100)
    np.random.shuffle(indices)

    # number of folds is 4
    k = len(names)//4
    namesfold0 = names[indices[0:k]]
    namesfold1 = names[indices[k:2*k]]
    namesfold2 = names[indices[2*k:3*k]]
    namesfold3 = names[indices[3*k:]]

    # training and validation folds
    trainingfold = {}
    validationfold = {}
    for i in range(4):
        ls = [namesfold0, namesfold1, namesfold2, namesfold3]
        valnames = ls.pop(i)
        trnames = np.hstack(ls)
        trainingfold[f'fold_{i}'] = list(trnames)
        validationfold[f'fold_{i}'] = list(valnames)

    cvdict = {'train': trainingfold, 'val': validationfold}
    with open(os.path.join(basepath, 'cross_validation.json'), 'w') as f:
        json.dump(cvdict, f, indent=4)
    return

if __name__ == '__main__':
    make_cvjson()