#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 01:29:20 2019

@author: xiezhihua
"""

import scipy.ndimage as nd
import numpy as np
import matplotlib.pyplot as plt

im = nd.imread('2.jpg', True)
im = im.astype('int32')
dx = nd.sobel(im,1)
dy = nd.sobel(im,1)
mag = np.hypot(dx,dy)
mag *= 255.0/np.max(mag) 

fig, ax = plt.subplots()
ax.imshow(mag, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.show()