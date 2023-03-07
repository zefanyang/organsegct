#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/7/2021 11:46 AM
# @Author: yzf
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(7, 5))
x = np.linspace(-5, 5)
y1 = np.exp(x)
y2 = x

axes[0].plot(x, y1, label='exponential mapping')
axes[0].plot(x, y2, label='linear mapping')
axes[0].set_title('Linear y axis')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(x, y1, label='exponential mapping')
axes[1].plot(x, y2, label='linear mapping')
axes[1].set_yscale('log')
axes[1].set_title('Logarithmic y axis')
axes[1].grid(True)
axes[1].legend()

plt.suptitle('$y_1 = e^{(x)}$ and $y_2 = x$')
plt.savefig('./intuition_of_logarithmic_scale_axis.jpg')
plt.show()
