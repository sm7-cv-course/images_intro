
#def hartleyRectify(points1, points2, imgSize, M1, M2, D1, D2, F = None):
#    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 3, 0.99)
#    #print 'mask\n', mask
#    retval, H1, H2 = cv2.stereoRectifyUncalibrated(
#        points1, points2, F, imgSize)
#    retval, M1i = cv2.invert(M1); retval, M2i = cv2.invert(M2)
#    R1, R2 = np.dot(np.dot(M1i, H1), M1), np.dot(np.dot(M2i, H2), M2)
#    map1x, map1y = cv2.initUndistortRectifyMap(M1, D1, R1, M1, imgSize, cv2.CV_32FC1)
#    map2x, map2y = cv2.initUndistortRectifyMap(M2, D2, R2, M2, imgSize, cv2.CV_32FC1)
#    return (map1x, map1y, map2x, map2y), F

# Source: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
import argparse
import cv2 as cv
import numpy as np
import glob

# Number of intricics corners. Important: must be 1 odd and 1 even number!
#bsize = (8, 5)
#bsize = (7, 7)
bsize = (7, 4)
scale=0.5

def read_points(fname):
    points = []

    # Using readlines()
    file = open(fname, 'r')
    Lines = file.readlines()

    count = 0
    # Strips the newline character
    for line in Lines:
        count += 1
        pts = line.split(',')
        points.append([np.float32(pts[0]), np.float32(pts[1])])
    return np.reshape(points, (len(points), 1, 2))


def readAllStereoPoints(ppath):
    pointsL = []
    pointsR = []

    # Using readlines()
    file = open(ppath, 'r')
    Lines = file.readlines()

    count = 0
    # Strips the newline character
    for line in Lines:
        count += 1
        pts = line.split(',')
        pointsL.append([np.float32(pts[0].replace("[","")), np.float32(pts[1].replace("]",""))])
        pointsR.append([np.float32(pts[2].replace("[","")), np.float32(pts[3].replace("];",""))])

    return np.asarray(pointsL), np.asarray(pointsR)

def read_camera_matrix(fpath):
    with np.load(fpath) as my_archive_file:
        matrix=my_archive_file["matrix"]
        coeffs=my_archive_file["dcoeffs"]
        return matrix, coeffs

def readFundamentalMatrix(mpath):
    file = open(mpath, 'r')
    Lines = file.readlines()

    F = np.zeros((3,3))

    i = 0
    # Strips the newline character
    for line in Lines:
        coeffs = line.split(';')
        j = 0
        for c in coeffs:
            F[i, j] = c
            j += 1
        i += 1
    return F


def get_images_sub_paths(folder_path):
    images = glob.glob(folder_path + '/*.jpg')
    images.extend(glob.glob(folder_path + '/*.png'))
    images.extend(glob.glob(folder_path + '/*.tiff'))
    images.extend(glob.glob(folder_path + '/*.tif'))
    images.extend(glob.glob(folder_path + '/*.gif'))
    images.extend(glob.glob(folder_path + '/*.jpeg'))
    return images


ap = argparse.ArgumentParser()
ap.add_argument("-a", "--allpoints", required=True, help="Path to all precalculated points from both left and right images.")
#ap.add_argument("-l", "--left", required=True, help="Path to precalculated points from left images.")
#ap.add_argument("-r", "--right", required=True, help="Path to precalculated points from right images.")
ap.add_argument("-m", "--matrixLeft", required=True, help="Path to camera matrix and distortion coefficients for left camera.")
ap.add_argument("-n", "--matrixRight", required=True, help="Path to camera matrix and distortion coefficients for right camera.")
ap.add_argument("-i", "--limage", required=True, help="Path to left image to apply undistort methods.")
ap.add_argument("-j", "--rimage", required=True, help="Path to right image to apply undistort methods.")
ap.add_argument("-p", "--lsource", required=True, help="Path to folder with all left images.")
ap.add_argument("-q", "--rsource", required=True, help="Path to folder with all right images.")
ap.add_argument("-o", "--output", required=False, help="Path to output folder (rectified stereo pair).")
ap.add_argument("-f", "--loadF", required=False, help="Path to file with fundamental matrix.")
args = vars(ap.parse_args())

input_path = './../images/calibration'

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((bsize[1]*bsize[0],3), np.float32)
objp[:,:2] = np.mgrid[0:bsize[0],0:bsize[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints_l = [] # 2d points in image plane.
imgpoints_r = []


# Load precalculated corresponding points
#if args["left"] is not None and args["right"] is not None:
#    input_path_l = args["left"]
#    input_path_r = args["right"]
#    pts_files_l = glob.glob(input_path_l + '/*.txt')
#    pts_files_r = glob.glob(input_path_r + '/*.txt')
#    pts_files_l.sort()
#    pts_files_r.sort()
#    #for pts_files in zip(pts_files_l, pts_files_r)
#    for fname in pts_files_l:
#        print(fname)
#        points = read_points(fname)
#        if len(points) == bsize[0] * bsize[1]:
#            imgpoints_l.append(points)
#            objpoints.append(objp)

#    for fname in pts_files_r:
#        print(fname)
#        points = read_points(fname)
#        if len(points) == bsize[0] * bsize[1]:
#            imgpoints_r.append(points)

if args["allpoints"] is not None:
    input_path = args["allpoints"]
    l_imgpointsAllTogether, r_imgpointsAllTogether = readAllStereoPoints(input_path)


if args["limage"] is not None:
    img_l = cv.imread(args["limage"])
    gray_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)

if args["rimage"] is not None:
    img_r = cv.imread(args["rimage"])
    gray_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

# Load Camera matrix and distortion coefficients
if args["matrixLeft"] is not None and args["matrixRight"] is not None:
    # opening the archive and accessing each array by name
    with np.load(args["matrixLeft"]) as matrixLeftFile:
        matrixL = matrixLeftFile["CameraMatrix"]
        dcoeffsL = matrixLeftFile["DistCoeffs"]
    with np.load(args["matrixRight"]) as matrixRightFile:
        matrixR = matrixRightFile["CameraMatrix"]
        dcoeffsR = matrixRightFile["DistCoeffs"]

#retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F =\
#cv.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, matrixL, dcoeffsL,
#matrixR, dcoeffsR, gray_l.shape[::-1])

#R1 = np.zeros(shape=(3,3))
#R2 = np.zeros(shape=(3,3))
#P1 = np.zeros(shape=(3,3))
#P2 = np.zeros(shape=(3,3))

#(width, height) = gray_l.shape[::-1]

#R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (width, height), R, T, R1, R2, P1, P2, Q=None, flags=cv.cv2.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0,0))

# Load or calculate Fundamental matrix
if args["loadF"] is not None:
    F = readFundamentalMatrix(args["loadF"])
#else:

#l_imgpointsAllTogether = np.zeros(shape=(len(imgpoints_l) * imgpoints_l[0].shape[0],\
#imgpoints_l[0].shape[1], imgpoints_l[0].shape[2]))
#r_imgpointsAllTogether = l_imgpointsAllTogether.copy()

#i = 0
#for pts_l, pts_r in zip(imgpoints_l, imgpoints_r):
#    j = 0
#    for pt_l, pt_r in zip(pts_l, pts_r):
#        l_imgpointsAllTogether[i * pts_l.shape[0] + j] = pt_l
#        r_imgpointsAllTogether[i * pts_l.shape[0] + j] = pt_r
#        j += 1
#    i += 1

#retval, H1, H2 = cv.stereoRectifyUncalibrated(imgpoints_l, imgpoints_r, F, img_l.shape[0:2])
retval, H1, H2 = cv.stereoRectifyUncalibrated(l_imgpointsAllTogether, r_imgpointsAllTogether, F, img_l.shape[0:2])


#map1_l, map2_l = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1,
#                                                 (width,height), cv.CV_16SC2)
#map1_r, map2_r  = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2,
#                                                      (width,height), cv.CV_16SC2)

# Show points on images
#if args["lsource"] is not None and args["rsource"] is not None:
#    input_path = args["lsource"]
#    l_images = get_images_sub_paths(input_path)
#    input_path = args["rsource"]
#    r_images = get_images_sub_paths(input_path)
#    l_images.sort()
#    r_images.sort()
#    count = 0
#    for l_imageName, r_imageName in zip(l_images, r_images):
#        print(l_imageName, r_imageName)
#        img_l = cv.imread(l_imageName)
#        img_r = cv.imread(r_imageName)
#        #gray_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
#        #gray_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
#        #l_small = cv.resize(gray_l, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
#        #r_small = cv.resize(gray_r, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
#        vis = cv.hconcat([img_l, img_r])

#        for pt_l, pt_r in zip(imgpoints_l[count], imgpoints_r[count]):
#            new_pt_r = pt_r + (gray_l.shape[1], 0)
#            vis = cv.line(vis, (int(pt_l[0][0]), int(pt_l[0][1])),\
#             (int(new_pt_r[0][0]), int(new_pt_r[0][1])), (0, 255, 0), thickness=1)

#        vis_small = cv.resize(vis, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
#        cv.imshow('vis', vis_small);
#        #cv.imshow('l_small', l_small);
#        #cv.imshow('r_small', r_small);
#        key = cv.waitKey(5) & 0xFF
#        if key == ord("q"):
#            break
#        count = count + 1
#        if count >= len(imgpoints_l):
#            break


#img_left = cv.remap(gray_l, map1_l, map2_l, cv.cv2.INTER_LINEAR)
#img_right = cv.remap(gray_r, map1_r, map2_r, cv.cv2.INTER_LINEAR)
img_left = gray_l#= cv.warpPerspective(gray_l, H1, img_l.shape[0:2])
w, h = img_l.shape[0:2]
img_right = cv.warpPerspective(gray_r, np.linalg.inv(np.mat(H1)) * np.mat(H2), (w,h))
#img_right = cv.warpPerspective(gray_r, H2, img_l.shape[0:2])

left_small = cv.resize(img_left, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
right_small = cv.resize(img_right, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
cv.imshow('left', left_small);
cv.imshow('right', right_small);

if args["output"] is not None:
    cv.imwrite(args["output"] + "/rectified_l.png", img_left)
    cv.imwrite(args["output"] + "/rectified_r.png", img_right)

key = cv.waitKey(50000) & 0xFF
if key == ord("q"):
    cv.destroyAllWindows()

cv.destroyAllWindows()
