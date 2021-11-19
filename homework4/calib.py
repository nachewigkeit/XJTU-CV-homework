import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

        cv2.drawChessboardCorners(img, shape, corners, ret)
        plt.imshow(img)
        plt.savefig(r"data/result/corner.png", bbox_inches='tight')
    else:
        print(fname)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx)  # 内参数矩阵
print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
