import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def gaus_noise(image, sigma):
    image = image / 255
    noise = np.random.normal(0, sigma, image.shape)
    out = image + noise
    out = np.clip(out, 0, 1)
    out = np.uint8(out * 255)
    return out


def match(image1, image2, method=cv.RANSAC):
    sift = cv.SIFT_create()
    #  使用SIFT查找关键点key points和描述符descriptors
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    matcher = cv.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m1, m2 in raw_matches:
        #  如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
        if m1.distance / m2.distance < 0.85:
            good_matches.append([m1])

    if len(good_matches) > 4:
        ptsA = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, status = cv.findHomography(ptsA, ptsB, method)
        imgOut = cv.warpPerspective(image2, H, (image1.shape[1], image1.shape[0]),
                                    flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

        plt.xticks([])
        plt.yticks([])
        plt.imshow(imgOut)
        print(np.sum(imgOut != (0, 0, 0)))
    else:
        print("Not Enough Match Pair")
