#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/9/2021 4:31 PM
# @Author: yzf
import json
import pandas as pd
cv_json = '/data/yzf/dataset/Project/ranet-dataset/cv_high_resolution.json'

with open(cv_json, 'r') as f:
    cv_dict = json.load(f)

cv_df = pd.DataFrame(data=cv_dict)

val_f0 = cv_df['val']['fold_0']
val_f1 = cv_df['val']['fold_1']
val_f2 = cv_df['val']['fold_2']
val_f3 = cv_df['val']['fold_3']

for ls in [val_f0, val_f1, val_f2, val_f3]:
    print(len(ls))
pass