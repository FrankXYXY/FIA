# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:21:23 2019

@author: Yi Xie
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

# Determine the slope by linear regression
def estimate_coef(x, y):
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return(b_0, b_1)

def plot_regression_line(x, y, b):
    plt.scatter(x, y, color = "m",marker = "o", s = 30)
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color = "g")
    plt.title("Linear Regression Plot ")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("a.jpg")

img = cv2.imread('2.jpg', 0)
blurred_img = cv2.medianBlur(img, 11)

s_mask = 17

sobelx = np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=s_mask))
b_sobelx = np.abs(cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=s_mask))
sobelx = interval_mapping(sobelx, np.min(sobelx), np.max(sobelx), 0, 255)
b_sobelx = interval_mapping(b_sobelx, np.min(sobelx), np.max(sobelx), 0, 255)

sobely = np.abs(cv2.Sobel(img,cv2.CV_64F,0,1,ksize=s_mask))
sobely = interval_mapping(sobely, np.min(sobely), np.max(sobely), 0, 255)
b_sobely = np.abs(cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=s_mask))
b_sobely = interval_mapping(b_sobely, np.min(sobely), np.max(sobely), 0, 255)

sobel_xy = 0.5 * sobelx + 0.5 * sobely
b_sobel_xy = 0.5 * b_sobelx + 0.5 * b_sobely

fig = plt.figure(figsize=(10, 14))
#plt.subplot(3,2,1),plt.imshow(sobelx,cmap = 'gray')
#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(3,2,2),plt.imshow(b_sobelx,cmap = 'gray')
#plt.title('Blurred Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(3,2,3),plt.imshow(sobely,cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#plt.subplot(3,2,4),plt.imshow(b_sobely,cmap = 'gray')
#plt.title('Blurred Sobel Y'), plt.xticks([]), plt.yticks([])
#plt.subplot(3,2,5),plt.imshow(sobel_xy,cmap = 'gray')
#plt.title('Sobel XY'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,6),plt.imshow(b_sobel_xy,cmap = 'gray')
plt.title('Blurred Sobel XY'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig("b.jpg")

if __name__ == "__main__":
  #  plot_regression_line(line_x,line_y,(intersection,slope))
    img = cv2.imread('b.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img,(5,5))
    # create a temporary depth image
    depth = [[0] * len(img[0])] * len(img)
    # the depth information should be a 2-d list
    line_data = []
    for i in range(len(img[0])):
        for j in range(len(img)):
            if img[j][i] > 30:
                # img[j][i] = 200
                line_data.append(len(img)-j)
                break
    line_data = np.array(line_data)
    mean = np.mean(line_data, axis=0)
    std = np.std(line_data, axis=0)
    line_y = np.array([e for e in line_data if (mean - 2 * std < e < mean + 2 * std)])  # Remove Outlier
    line_x = np.linspace(1, len(line_y), len(line_y))
    intersection, slope = estimate_coef(line_x, line_y)
    theta = np.arctan(slope)
    plot_regression_line(line_x,line_y,(intersection,slope))
     
    lst = []
    for i in range(int(0.45 * len(img)), int(0.55 * len(img))):
        diameter = []
        for j in range(len(img[i])):
            index = j
            while img[i][index] < 50:
                img[i][index] = 200
                index += 1
                if index >= len(img[i]):
                    break
            if index > j:
                # distance = (depth[i][index] + depth[i][j]) / 2
                diameter.append(index - j)
        lst.append(diameter)
    output = []
    index = min([len(x) for x in lst])
    for j in range(index):
        temp = 0
        k = 0
        for i in range(len(lst)):
            temp += lst[i][j]
            k += 1
        output.append(np.cos(theta) * np.round(temp/k, 4))
        # output.append((np.cos(theta) * np.round(temp/k,4), distance[i]))
    print(output)
    print(theta * 180 / np.pi)
    cv2.imwrite("output.jpg", img)