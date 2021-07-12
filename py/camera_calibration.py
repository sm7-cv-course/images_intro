# Source: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
import argparse
import cv2 as cv
import numpy as np
import glob

# Number of intricics corners. Important: must be 1 odd and 1 even number!
#bsize = (8, 5)
bsize = (7, 4)

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Path to source images.")
ap.add_argument("-i", "--image", required=False, help="Path to image to apply undistort methods.")
args = vars(ap.parse_args())

input_path = './../images/calibration'
if args["source"] is not None:
    input_path = args["source"]

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((bsize[1]*bsize[0],3), np.float32)
objp[:,:2] = np.mgrid[0:bsize[0],0:bsize[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(input_path + '/*.jpg')
images.extend(glob.glob(input_path + '/*.png'))
images.extend(glob.glob(input_path + '/*.tiff'))
images.extend(glob.glob(input_path + '/*.gif'))
images.extend(glob.glob(input_path + '/*.jpeg'))


for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, bsize, None)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, bsize, corners2, ret)
        cv.imshow('img', img)
        key = cv.waitKey(500) & 0xFF
        if key == ord("q"):
            break
cv.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if args["image"] is not None:
    input_image_path = args["image"]

    img = cv.imread(input_image_path)
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite('calibresult.png', dst)

    # undistort - II
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite('calibresult2.png', dst)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error/len(objpoints)))

