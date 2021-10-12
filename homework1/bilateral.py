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
gausFilter = utils.GausFilter2d((size, size), sigma1)

bilateralImage1 = utils.bilateralConv(image, gausFilter, 1, "reflect", (sigma, sigma, sigma, sigma))
bilateralImage2 = utils.bilateralConv(image, gausFilter, 3e-1, "reflect", (sigma, sigma, sigma, sigma))
bilateralImage3 = utils.bilateralConv(image, gausFilter, 1e-1, "reflect", (sigma, sigma, sigma, sigma))
bilateralImage4 = utils.bilateralConv(image, gausFilter, 1e-2, "reflect", (sigma, sigma, sigma, sigma))

plt.figure(figsize=(20, 20))
plt.subplot(221)
plt.title("sigma:1", fontsize=24)
plt.imshow(bilateralImage1)
plt.subplot(222)
plt.title("sigma:0.3", fontsize=24)
plt.imshow(bilateralImage2)
plt.subplot(223)
plt.title("sigma:0.1", fontsize=24)
plt.imshow(bilateralImage3)
plt.subplot(224)
plt.title("sigma:0.01", fontsize=24)
plt.imshow(bilateralImage4)
plt.savefig(r"image/bilateral.png")
plt.show()
