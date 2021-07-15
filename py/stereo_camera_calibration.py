# Source: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
import argparse
import cv2 as cv
import numpy as np
import glob

# Number of intricics corners. Important: must be 1 odd and 1 even number!
#bsize = (8, 5)
bsize = (7, 7)
#bsize = (7, 4)
scale=0.25

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
        #print("Line{}: {}".format(count, line.strip()))

    return np.reshape(points, (len(points), 1, 2))


def read_camera_matrix(fpath):
    with np.load(fpath) as my_archive_file:
        matrix=my_archive_file["matrix"]
        coeffs=my_archive_file["dcoeffs"]
        return matrix, coeffs


ap = argparse.ArgumentParser()
ap.add_argument("-l", "--left", required=True, help="Path to precalculated points from left images.")
ap.add_argument("-r", "--right", required=True, help="Path to precalculated points from right images.")
ap.add_argument("-m", "--matrixLeft", required=True, help="Path to camera matrix and distortion coefficients for left camera.")
ap.add_argument("-n", "--matrixRight", required=True, help="Path to camera matrix and distortion coefficients for right camera.")
ap.add_argument("-i", "--limage", required=True, help="Path to left image to apply undistort methods.")
ap.add_argument("-j", "--rimage", required=True, help="Path to right image to apply undistort methods.")
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
if args["left"] is not None and args["right"] is not None:
    input_path_l = args["left"]
    input_path_r = args["right"]
    pts_files_l = glob.glob(input_path_l + '/*.txt')
    pts_files_r = glob.glob(input_path_r + '/*.txt')
    pts_files_l.sort()
    pts_files_r.sort()
    #for pts_files in zip(pts_files_l, pts_files_r)
    for fname in pts_files_l:
        print(fname)
        points = read_points(fname)
        if len(points) == bsize[0] * bsize[1]:
            imgpoints_l.append(points)
            objpoints.append(objp)

    for fname in pts_files_r:
        print(fname)
        points = read_points(fname)
        if len(points) == bsize[0] * bsize[1]:
            imgpoints_r.append(points)

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

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F =\
cv.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, matrixL, dcoeffsL,
matrixR, dcoeffsR, gray_l.shape[::-1])

R1 = np.zeros(shape=(3,3))
R2 = np.zeros(shape=(3,3))
P1 = np.zeros(shape=(3,3))
P2 = np.zeros(shape=(3,3))

(width, height) = gray_l.shape[::-1]

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (width, height), R, T, R1, R2, P1, P2, Q=None, flags=cv.cv2.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0,0))

#
map1_l, map2_l = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1,
                                                 (width,height), cv.CV_16SC2)
map1_r, map2_r  = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2,
                                                       (width,height), cv.CV_16SC2)

img_left = cv.remap(gray_l, map1_l, map2_l, cv.cv2.INTER_LINEAR);
img_right = cv.remap(gray_r, map1_r, map2_r, cv.cv2.INTER_LINEAR)
left_small = cv.resize(img_left, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
right_small = cv.resize(img_right, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
cv.imshow('left', left_small);
cv.imshow('right', right_small);

key = cv.waitKey(50000) & 0xFF
if key == ord("q"):
    cv.destroyAllWindows()

cv.destroyAllWindows()
#double cv::stereoCalibrate	(	InputArrayOfArrays 	objectPoints,
#InputArrayOfArrays 	imagePoints1,
#InputArrayOfArrays 	imagePoints2,
#InputOutputArray 	cameraMatrix1,
#InputOutputArray 	distCoeffs1,
#InputOutputArray 	cameraMatrix2,
#InputOutputArray 	distCoeffs2,
#Size 	imageSize,
#InputOutputArray 	R,
#InputOutputArray 	T,
#OutputArray 	E,
#OutputArray 	F,
#OutputArray 	perViewErrors,
#int 	flags = CALIB_FIX_INTRINSIC,
#TermCriteria 	criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6)
#)


#if args["image"] is not None:
#    img = cv.imread(args["image"])
#    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#    # Estimate Error
#    h, w = img.shape[:2]
#    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

#    # undistort
#    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
#    # crop the image
#    x, y, w, h = roi
#    dst = dst[y:y + h, x:x + w]
#    cv.imwrite('calibresult.png', dst)

#    # undistort - II
#    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
#    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
#    # crop the image
#    x, y, w, h = roi
#    dst = dst[y:y + h, x:x + w]
#    cv.imwrite('calibresult2.png', dst)

#    mean_error = 0
#    for i in range(len(objpoints)):
#        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
#        mean_error += error
#    print("total error: {}".format(mean_error/len(objpoints)))


#def read_camera_matrix(fpath)
#    file = open(fname, 'r')
#    Lines = file.readlines()

#    matrix =  []
#    count = 0
#    # Strips the newline character
#    for line in Lines:
#        if i < 3:
#            pts = line.split(',')
#            matrix.append([np.float32(pts[0]), np.float32(pts[1]), np.float32(pts[2])])
#        else:
#            dcoeffs = [np.float32(pts[0]), np.float32(pts[1]), np.float32(pts[2])]
#        count += 1
