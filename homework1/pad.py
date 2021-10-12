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

padImage1 = utils.padding(image, "zero", (sigma, sigma, sigma, sigma))
padImage2 = utils.padding(image, "wrap", (sigma, sigma, sigma, sigma))
padImage3 = utils.padding(image, "reflect", (sigma, sigma, sigma, sigma))
padImage4 = utils.padding(image, "copy", (sigma, sigma, sigma, sigma))

gausImage1 = utils.conv(image, gausFilter1, "zero", (sigma, sigma, sigma, sigma))
gausImage2 = utils.conv(image, gausFilter1, "wrap", (sigma, sigma, sigma, sigma))
gausImage3 = utils.conv(image, gausFilter1, "reflect", (sigma, sigma, sigma, sigma))
gausImage4 = utils.conv(image, gausFilter1, "copy", (sigma, sigma, sigma, sigma))

plt.figure(figsize=(40, 20))
plt.subplot(241)
plt.title("zero", fontsize=24)
plt.imshow(padImage1)
plt.subplot(242)
plt.title("wrap", fontsize=24)
plt.imshow(padImage2)
plt.subplot(243)
plt.title("reflect", fontsize=24)
plt.imshow(padImage3)
plt.subplot(244)
plt.title("copy", fontsize=24)
plt.imshow(padImage4)
plt.subplot(245)
plt.imshow(gausImage1)
plt.subplot(246)
plt.imshow(gausImage2)
plt.subplot(247)
plt.imshow(gausImage3)
plt.subplot(248)
plt.imshow(gausImage4)
plt.savefig(r"image/pad.png")
plt.show()
