import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./../images/IMG_1756.JPG')

img_grey = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

# Create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl_r = clahe.apply(img[...,0])
cl_g = clahe.apply(img[...,1])
cl_b = clahe.apply(img[...,2])

cl_rgb = cv2.merge((cl_r, cl_g, cl_b))

cv2.imwrite('./../out/clahe_2.jpg',cl_rgb)
