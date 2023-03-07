#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 3/18/2021 3:46 PM
# @Author: yzf
"""Using pandas to read cv.jason"""

import pandas as pd

j = '/data/yzf/dataset/Project/ranet-dataset/cv_high_resolution.json'

df = pd.read_json(j)

# df.to_csv('./cv.csv')

pass