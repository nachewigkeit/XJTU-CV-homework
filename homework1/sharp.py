import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import utils


def getImage(sigma2):
    sigma1 = 10
    d1 = sigma1 * sigma1
    d2 = sigma2 * sigma2
    size = 2 * sigma1 + 1
    gausFilter2d1 = utils.GausFilter2d((size, size), ((d1, 0), (0, d1)))
    gausFilter2d2 = utils.GausFilter2d((size, size), ((d2, 0), (0, d2)))

    dogFilter = gausFilter2d2 - gausFilter2d1
    dogImage = utils.conv(image, dogFilter, "reflect", (sigma1, sigma1, sigma1, sigma1))
    dogImage = dogImage[:, :, 0]

    sharpFilter = np.zeros((size, size))
    sharpFilter[(size - 1) // 2, (size - 1) // 2] = 1
    sharpFilter = sharpFilter + 3 * dogFilter
    sharpImage = utils.conv(image, sharpFilter, "reflect", (sigma1, sigma1, sigma1, sigma1))

    dogFilter = (dogFilter - dogFilter.min()) / (dogFilter.max() - dogFilter.min())
    dogImage = (dogImage - dogImage.min()) / (dogImage.max() - dogImage.min())

    return dogFilter, dogImage, sharpImage


image = Image.open(r"image/origin.jpg")
image = np.array(image)
image = image / 255

dogFilter1, dogImage1, sharpImage1 = getImage(1)
dogFilter2, dogImage2, sharpImage2 = getImage(3)
dogFilter3, dogImage3, sharpImage3 = getImage(5)

plt.figure(figsize=(30, 30))
plt.subplot(331)
plt.title("sigma:1-10", fontsize=24)
plt.imshow(dogFilter1, cmap="gray")
plt.subplot(332)
plt.title("sigma:3-10", fontsize=24)
plt.imshow(dogFilter2, cmap="gray")
plt.subplot(333)
plt.title("sigma:5-10", fontsize=24)
plt.imshow(dogFilter3, cmap="gray")
plt.subplot(334)
plt.imshow(dogImage1, cmap="gray")
plt.subplot(335)
plt.imshow(dogImage2, cmap="gray")
plt.subplot(336)
plt.imshow(dogImage3, cmap="gray")
plt.subplot(337)
plt.imshow(sharpImage1)
plt.subplot(338)
plt.imshow(sharpImage2)
plt.subplot(339)
plt.imshow(sharpImage3)
plt.savefig(r"image/sharp.png")
plt.show()
