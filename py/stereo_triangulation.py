import argparse
import cv2 as cv
import numpy as np
import glob
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

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


def read_all_matrices(fpath):
    with np.load(fpath) as my_archive_file:
        R=my_archive_file["RotMatrix"]
        T=my_archive_file["Tvector"]
        R1=my_archive_file["Rot1"]
        R2=my_archive_file["Rot2"]
        P1=my_archive_file["Proj1"]
        P2=my_archive_file["Proj2"]
        Q=my_archive_file["DispToDepth"]
        map1_l=my_archive_file["Map1L"]
        map2_l=my_archive_file["Map2L"]
        map1_r=my_archive_file["Map1R"]
        map2_r=my_archive_file["Map2R"]

        return R,T,R1,R2,P1,P2,Q,map1_l,map2_l,map1_r,map2_r

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--allpoints", required=True, help="Path to all precalculated points from both left and right images.")
ap.add_argument("-q", "--loadDepthMatrix", required=True, help="Path to file with all matrices.")
args = vars(ap.parse_args())


#if args["loadF"] is not None:
#   F = readFundamentalMatrix(args["loadF"])

#if args["allpoints"] is not None:
#    input_path = args["allpoints"]
#    l_imgpointsAllTogether, r_imgpointsAllTogether = readAllStereoPoints(input_path)

if args["loadDepthMatrix"] is not None:
    R,T,R1,R2,P1,P2,Q,map1_l,map2_l,map1_r,map2_r = read_all_matrices(args["loadDepthMatrix"])

allPoints4D = []
allXYZ1 = []

fig = plt.figure()
ax = plt.axes(projection='3d')
ax = plt.axes(projection='3d')

if args["allpoints"] is not None:
    input_path = args["allpoints"]
    pts_files = glob.glob(input_path + '/*.txt')
    for fname in pts_files:
        l_imgPts, r_imgPts = readAllStereoPoints(fname)
        if l_imgPts.shape[0] == 0 or r_imgPts.shape[0] == 0:
            continue
        points4D = cv.triangulatePoints(P1, P2, np.transpose(l_imgPts), np.transpose(r_imgPts))
        XYZ1 = points4D / points4D[3]
        allPoints4D.append(points4D)
        allXYZ1.append(XYZ1)
        ax.scatter(XYZ1[0],XYZ1[1],XYZ1[2])
        plt.pause(0.05)
plt.show()



# https://stackoverflow.com/questions/22667121/pointcloud-from-two-undistorted-images
#Mat essential = cam_matrix1.t() * F * cam_matrix1;


