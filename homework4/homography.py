from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt

homo = loadmat(r"data/homography.mat")
H = homo['H_1']

img = cv2.imread(r"image/001.bmp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgOut = cv2.warpPerspective(gray, H, (3 * gray.shape[1], 3 * gray.shape[0]),
                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.title("Original", fontsize=36)
plt.xticks([])
plt.yticks([])
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.title("Homography", fontsize=36)
plt.xticks([])
plt.yticks([])
plt.imshow(imgOut, cmap='gray')
plt.show()
