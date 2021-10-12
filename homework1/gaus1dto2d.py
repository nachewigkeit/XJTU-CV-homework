import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import utils

image = Image.open(r"image/origin.jpg")
image = np.array(image)
image = image / 255

sigma = 10
size = 2 * sigma + 1
gausFilter1d = utils.GausFilter1d(size, sigma)
gausFilter1d1 = gausFilter1d.reshape((1, -1))
gausFilter1d2 = gausFilter1d.reshape((-1, 1))

d = sigma * sigma
sigma2d = ((d, 0), (0, d))
gausFilter2d = utils.GausFilter2d((size, size), sigma2d)
gausFilter2dfrom1d = utils.conv(gausFilter1d1, gausFilter1d2, "zero", (0, size - 1, 0, size - 1)).squeeze()

gausImage1 = utils.conv(image, gausFilter1d1, "reflect", (1, sigma, 1, sigma))
gausImage2 = utils.conv(image, gausFilter2dfrom1d, "reflect", (sigma, sigma, sigma, sigma))

plt.figure(figsize=(30, 20))
plt.subplot(231)
plt.imshow(gausFilter1d1, cmap="gray")
plt.subplot(232)
plt.imshow(gausFilter1d2, cmap="gray")
plt.subplot(233)
plt.imshow(gausFilter2dfrom1d, cmap="gray")
plt.subplot(235)
plt.title("1d", fontsize=24)
plt.imshow(gausImage1)
plt.subplot(236)
plt.title("1d+1d=2d", fontsize=24)
plt.imshow(gausImage2)
plt.savefig(r"image/1dto2d.png")
plt.show()
