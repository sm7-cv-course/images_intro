import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False, help="Path to source image.")
ap.add_argument("-o", "--output", required=False, help="Path to output image.")
args = vars(ap.parse_args())

path = './../images/IMG_1756.JPG'
out_path = './../out/clahe_2.jpg'

if args["source"] is not None:
    path = args["source"]

if args["output"] is not None:
    out_path = args["output"]

img = cv2.imread(path)

img_grey = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

# Create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl_r = clahe.apply(img[..., 0])
cl_g = clahe.apply(img[..., 1])
cl_b = clahe.apply(img[..., 2])

cl_rgb = cv2.merge((cl_r, cl_g, cl_b))

cv2.imwrite(out_path, cl_rgb)
