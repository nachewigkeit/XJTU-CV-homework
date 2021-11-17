import glob
import cv2
import numpy as np

# 获取标定板角点的位置
shape = (7, 11)
objp = np.zeros((shape[0] * shape[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:shape[1], 0:shape[0]].T.reshape(-1, 2)

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


        print(objp.shape)
        print(corners2.shape)
        H, status = cv2.findHomography(objp[:, :2], corners2.squeeze(), cv2.RANSAC)
        imgOut = cv2.warpPerspective(gray, H, (gray.shape[1], gray.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        cv2.imshow('img', imgOut)
        cv2.waitKey(-1)

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
print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
print("tvecs:\n", tvecs)  # 平移向量  # 外参数
