import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./../images/Fig0203(a)(chest-xray).tif')

hist,bins = np.histogram(img.flatten(), 256, [0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# Plot the input
plt.subplot(2, 2, 1), plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.title('Histogram + CDF'), plt.xticks([]), plt.yticks([])

# Equalization
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
img2 = cdf[img]

# Plot the results
plt.subplot(2,2,3),plt.imshow(img2, cmap = 'gray')
plt.title('Equalized image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4), plt.plot(cdf, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.title('Histogram + CDF'), plt.xticks([]), plt.yticks([])
plt.show()

# Histogram equalization by OpenCV
img_grey = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
equ = cv2.equalizeHist(img_grey)
res = np.hstack((img_grey, equ)) #stacking images side-by-side
cv2.imwrite('./../out/hist_eq_res.png', res)
