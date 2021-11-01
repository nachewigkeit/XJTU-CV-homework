from scipy.ndimage import sobel, gaussian_filter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def NMS(corner, thres):
    answer = np.zeros((corner.shape[0], corner.shape[1]))
    for i in range(1, corner.shape[0] - 1):
        for j in range(1, corner.shape[1] - 1):
            if corner[i, j] > thres and corner[i, j] == corner[i - 1:i + 2, j - 1:j + 2].max():
                answer[i, j] = 1

    return answer


img = Image.open(r"image\cake.jpg")
img = img.convert('L')
img = np.array(img, dtype='float32') / 255

xEdge = sobel(img, 0)
yEdge = sobel(img, 1)

sigma = 5
xx = gaussian_filter(xEdge * xEdge, sigma)
xy = gaussian_filter(xEdge * yEdge, sigma)
yy = gaussian_filter(yEdge * yEdge, sigma)

det = xx * yy - xy * xy
tr = xx + yy
corner = det / tr
corner /= corner.max()
answer = NMS(corner, 0.2)

plt.imshow(corner, cmap='gray')
plt.show()
plt.imshow(answer, cmap='gray')
plt.show()
