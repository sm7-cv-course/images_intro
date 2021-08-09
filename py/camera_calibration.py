# Source: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
import argparse
import cv2 as cv
import numpy as np
import glob

# Number of intricics corners. Important: must be 1 odd and 1 even number!
#bsize = (8, 5)
#bsize = (7, 7)
bsize = (7, 4)
scale = 0.5

def read_points(fname):
    points = []

    # Using readlines()
    file = open(fname, 'r')
    Lines = file.readlines()

    file.close()

    count = 0
    # Strips the newline character
    for line in Lines:
        count += 1
        pts = line.split(',')
        points.append([np.float32(pts[0]), np.float32(pts[1])])
        #print("Line{}: {}".format(count, line.strip()))

    return np.reshape(points, (len(points), 1, 2))


def save_points(filepath, pts):
    fp = open(filepath, 'w')

    for pt in pts:
        fp.write("{},{}\n".format(pt[0][0], pt[0][1]))

    fp.close()


def save_matrix(fpath, matrix, dcoeffs, saveTXT=True):
    np.savez(fpath, CameraMatrix=matrix, DistCoeffs=dcoeffs)
    if saveTXT:
        f = open(fpath, 'w')
    #for i in range(3): # ToDo: implement 3 matrices case writting!
        f.write("%s\n" % matrix)
        f.write("%s\n" % dcoeffs)
        f.close()


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False, help="Path to source images.")
ap.add_argument("-p", "--points", required=False, help="Path to precalculated corresponding points.")
ap.add_argument("-i", "--image", required=False, help="Path to image to apply undistort methods.")
ap.add_argument("-o", "--output", required=False, help="Path to output file with camera matrix and distortion coefficients.")
ap.add_argument("-w", "--write", required=False, help="Path to folder to write found corner points.")

ap.add_argument("-n", "--newobjpts", required=False, help="Path to folder to write new Object Points estimated with calibrateCameraRO.")
args = vars(ap.parse_args())

input_path = './../images/calibration'

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((bsize[1]*bsize[0],3), np.float32)
objp[:,:2] = np.mgrid[0:bsize[0],0:bsize[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

if args["source"] is not None:
    input_path = args["source"]
    images = glob.glob(input_path + '/*.jpg')
    images.extend(glob.glob(input_path + '/*.png'))
    images.extend(glob.glob(input_path + '/*.tiff'))
    images.extend(glob.glob(input_path + '/*.tif'))
    images.extend(glob.glob(input_path + '/*.gif'))
    images.extend(glob.glob(input_path + '/*.jpeg'))

# Find corresponding points in images
    for fname in images:
        print(fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #cv.imshow('gray', gray)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, bsize, None)
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, bsize, corners2, ret)
            img_small = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
            cv.imshow('img', img_small)
            key = cv.waitKey(500) & 0xFF
            if key == ord("q"):
                break
            # Save found corners points if needed
            if args["write"] is not None:
                ppath = args["write"] + '/' + fname.split('/')[-1].split(".")[0] + '.txt'
                save_points(ppath, corners)
#    rms = calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
#                            cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
#                            flags | CALIB_FIX_K3 | CALIB_USE_LU);
#calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint, cameraMatrix, distCoeffs[, rvecs[, tvecs[, newObjPoints[, flags[, criteria]]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints
    ret, mtx, dist, rvecs, tvecs, newObjPoints = cv.calibrateCameraRO(objpoints, imgpoints, gray.shape[::-1], 6, None, None)
    #ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    cv.destroyAllWindows()
    if args["output"] is not None:
        save_matrix(args["output"], mtx, dist)


# Load precalculated corresponding points
if args["points"] is not None:
    input_path = args["points"]
    pts_files = glob.glob(input_path + '/*.txt')
    for fname in pts_files:
        print(fname)
        points = read_points(fname)
        if len(points) == bsize[0] * bsize[1]:
            imgpoints.append(points)
            objpoints.append(objp)

    if args["image"] is not None:
        input_image_path = args["image"]
        img = cv.imread(input_image_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        ret, mtx, dist, rvecs, tvecs, newObjPoints = cv.calibrateCameraRO(objpoints, imgpoints, gray.shape[::-1], 6, None, None)
        newObjPoints_all = []
        ret, mtx, dist, rvecs, tvecs, newObjPoints = cv.calibrateCameraRO(objpoints, imgpoints, gray.shape[::-1], 6, mtx, dist, rvecs, tvecs, newObjPoints[0])
        for objpoint in objpoints:
            newObjPoints_all.append(newObjPoints[0])
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        if args["output"] is not None:
            save_matrix(args["output"], newcameramtx, dist)


if args["image"] is not None:
    input_image_path = args["image"]

    img = cv.imread(input_image_path)
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    if args["output"] is not None:
        save_matrix(args["output"], newcameramtx, dist)

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
        imgpoints2, _ = cv.projectPoints(newObjPoints_all[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error/len(objpoints)))

