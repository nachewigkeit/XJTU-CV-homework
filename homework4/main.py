import glob
import cv2
import numpy as np

# 获取标定板角点的位置
shape = (7, 11)
objp = np.zeros((shape[0] * shape[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:shape[0], 0:shape[1]].T.reshape(-1, 2)
obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob("image/*.bmp")
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, shape)

    if ret:
        obj_points.append(objp)

        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        img_points.append(corners2)

        '''
        cv2.drawChessboardCorners(img, shape, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(-1)
        '''
    else:
        print(fname)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx)  # 内参数矩阵
print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

img = cv2.imread(r"image/001.bmp")
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]

mean_error = 0

for j in range(5):
    dist_new = dist.copy()
    dist_new[0, j] = 0
    print(dist_new)
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist_new)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2)
        mean_error += error
    print(error / len(img_points))
