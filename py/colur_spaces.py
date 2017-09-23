import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('./../images/c130someairportRotate_small.bmp')

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
img_hsv_full = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV_FULL)

plt.imshow(img_gray, cmap = 'gray')
plt.show()

plt.subplot(2, 2, 1), plt.imshow(img_rgb)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(img_hsv[...,0], cmap = 'gray')
plt.title('Hue'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(img_hsv[...,1], cmap = 'gray')
plt.title('Saturation'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(img_hsv[...,2], cmap = 'gray')
plt.title('Value'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(2, 2, 1), plt.imshow(img_rgb)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(img_rgb[...,0],cmap = 'gray')
plt.title('Red'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(img_rgb[...,1],cmap = 'gray')
plt.title('Green'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(img_rgb[...,2],cmap = 'gray')
plt.title('Blue'), plt.xticks([]), plt.yticks([])
plt.show()

# Print out all possible flags for cvtColor
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)
