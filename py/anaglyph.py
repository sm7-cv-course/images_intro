import argparse
import cv2 as cv
import numpy as np
import glob

scale=0.25
colorScheme = 'optimized'
BaseShift = 420
AnaglyphShift = 20

# Anaglyph convertion matrices
#https://github.com/miguelgrinberg/anaglyph.py/blob/master/anaglyph.py
mixMatrices = {
'true': [ [ 0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 0, 0, 0.299, 0.587, 0.114 ] ],
'mono': [ [ 0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114 ] ],
'color': [ [ 1, 0, 0, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 1, 0, 0, 0, 1 ] ],
'halfcolor': [ [ 0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 1, 0, 0, 0, 1 ] ],
'optimized': [ [ 0, 0.7, 0.3, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 1, 0, 0, 0, 1 ] ],
}

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--limage", required=True, help="Path to left rectified image.")
ap.add_argument("-r", "--rimage", required=True, help="Path to right rectified image.")
ap.add_argument("-o", "--output", required=False, help="Path to output folder.")
args = vars(ap.parse_args())

if args["limage"] is not None:
    img_left = cv.imread(args["limage"])

if args["rimage"] is not None:
    img_right = cv.imread(args["rimage"])

# Cut images to shorten stereo pair base if needed
if BaseShift - AnaglyphShift != 0:
    img_left = img_left[:,(BaseShift - AnaglyphShift):img_left.shape[1],:]
    img_right = img_right[:,0:img_right.shape[1] - (BaseShift - AnaglyphShift),:]

if len(img_left.shape) == 2:
    resultArray = np.zeros((img_left.shape[0],img_left.shape[1],3))
    m = mixMatrices[colorScheme]
    resultArray[:,:,0] = img_left*m[0][6] + img_left*m[0][7] + img_left*m[0][8] + img_right*m[1][6] + img_right*m[1][7] + img_right*m[1][8]
    resultArray[:,:,1] = img_left*m[0][3] + img_left*m[0][4] + img_left*m[0][5] + img_right*m[1][3] + img_right*m[1][4] + img_right*m[1][5]
    resultArray[:,:,2] = img_left*m[0][0] + img_left*m[0][1] + img_left*m[0][2] + img_right*m[1][0] + img_right*m[1][1] + img_right*m[1][2]

elif len(img_left.shape) > 2:
    lb,lg,lr = cv.split(np.asarray(img_left[:,:]))
    rb,rg,rr = cv.split(np.asarray(img_right[:,:]))
    resultArray = np.zeros((img_left.shape[0],img_left.shape[1],3))
    m = mixMatrices[colorScheme]
    resultArray[:,:,0] = lb*m[0][6] + lg*m[0][7] + lr*m[0][8] + rb*m[1][6] + rg*m[1][7] + rr*m[1][8]
    resultArray[:,:,1] = lb*m[0][3] + lg*m[0][4] + lr*m[0][5] + rb*m[1][3] + rg*m[1][4] + rr*m[1][5]
    resultArray[:,:,2] = lb*m[0][0] + lg*m[0][1] + lr*m[0][2] + rb*m[1][0] + rg*m[1][1] + rr*m[1][2]

    if args["output"] is not None:
        cv.imwrite(args["output"] + "/anaglyph.png", resultArray)
