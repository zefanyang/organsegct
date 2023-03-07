#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 3/16/2021 2:21 PM
# @Author: yzf
"""Prototype: box plots of 8 organs' segmentation metrics for performance comparison"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

f1 = './dsc_ours.csv'
f2 = './dsc_unet.csv'
organs = ['spleen', 'left kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'pancreas', 'duodenum']
df1 = pd.read_csv(f1, index_col=0)
df2 = pd.read_csv(f2, index_col=0)
df1.rename(columns={str(k): v for k, v in enumerate(organs)}, inplace=True)
df2.rename(columns={str(k): v for k, v in enumerate(organs)}, inplace=True)

df1['algorithm'] = 'GRFE-Net'
df2['algorithm'] = 'U-Net'

df_wide = df1.append(df2)
df_long = df_wide.melt(id_vars='algorithm', value_vars=organs, var_name='organ', value_name='DSC')

# removing rows with zero values
# df_long.drop(index=df_long[df_long['DSC'] == 0].index, axis=0, inplace=True)

# removing rows with null values
# df_long.isnull().sum()
# col = pd.DataFrame(data={k: None for k in df_long.columns.values}, index=[0])  # for testing
# df_long.append(col)
# df_long.dropna()

# TODO: style

sns.boxplot(y='organ', x='DSC', hue='algorithm', data=df_long)
plt.show()



# data = {
#     'apple': [float('Nan'), 1., 2., 3.],
#     'oranges': [0., 1., 2., 3.],
# }
#
# purchases = pd.DataFrame(data, index=['a', 'b', 'c', 'd'])
#
# print(purchases)