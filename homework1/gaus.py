import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import utils

image = Image.open(r"image/origin.jpg")
image = np.array(image)
image = image / 255

sigma = 10
d = sigma * sigma
size = 2 * sigma + 1
sigma1 = ((d, 0), (0, d))
gausFilter1 = utils.GausFilter2d((size, size), sigma1)
sigma2 = ((d, 0), (0, 1))
gausFilter2 = utils.GausFilter2d((size, size), sigma2)
sigma3 = ((d, 0.9*d), (0.9*d, d))
gausFilter3 = utils.GausFilter2d((size, size), sigma3)

gausImage1 = utils.conv(image, gausFilter1, "reflect", (sigma, sigma, sigma, sigma))
gausImage2 = utils.conv(image, gausFilter2, "reflect", (sigma, sigma, sigma, sigma))
gausImage3 = utils.conv(image, gausFilter3, "reflect", (sigma, sigma, sigma, sigma))

plt.figure(figsize=(30, 20))
plt.subplot(231)
plt.imshow(gausFilter1, cmap="gray")
plt.subplot(232)
plt.imshow(gausFilter2, cmap="gray")
plt.subplot(233)
plt.imshow(gausFilter3, cmap="gray")
plt.subplot(234)
plt.imshow(gausImage1)
plt.subplot(235)
plt.imshow(gausImage2)
plt.subplot(236)
plt.imshow(gausImage3)
plt.savefig(r"image/gaus.png")
plt.show()
