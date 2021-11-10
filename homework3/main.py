import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

originImage = mpimg.imread("image/cake.jpg")
image1 = originImage[:300, :400, :]
image2 = originImage[-300:, -400:, :]

sift = cv.SIFT_create()
#  使用SIFT查找关键点key points和描述符descriptors
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)

kp_image1 = cv.drawKeypoints(image1, kp1, None)
kp_image2 = cv.drawKeypoints(image2, kp2, None)

plt.imshow(kp_image1)
plt.show()

plt.imshow(kp_image2)
plt.show()

ratio = 0.85
matcher = cv.BFMatcher()
raw_matches = matcher.knnMatch(des1, des2, k=2)
good_matches = []
for m1, m2 in raw_matches:
    #  如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
    if m1.distance < ratio * m2.distance:
        good_matches.append([m1])

matches = cv.drawMatchesKnn(image1, kp1, image2, kp2, good_matches, None, flags=2)

plt.figure()
plt.imshow(matches)
plt.show()

#  单应性矩阵有八个参数，每一个对应的像素点可以产生2个方程(x一个，y一个)，那么需要四个像素点就能解出单应性矩阵
if len(good_matches) > 4:
    #  计算匹配时间
    ptsA = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ptsB = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ransacReprojThreshold = 4
    #  单应性矩阵可以将一张图通过旋转、变换等方式与另一张图对齐
    H, status = cv.findHomography(ptsA, ptsB, cv.RANSAC, ransacReprojThreshold);
    imgOut = cv.warpPerspective(image2, H, (image1.shape[1], image1.shape[0]),
                                flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

    plt.figure()
    plt.imshow(imgOut)
    plt.show()
