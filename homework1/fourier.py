import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from PIL import Image

import utils

image = Image.open(r"image/origin.jpg")
image = np.array(image)
image = image / 255

sigma1 = 10
d1 = sigma1 * sigma1
gausFilter1 = utils.GausFilter2d((image.shape[0], image.shape[1]), ((d1, 0), (0, d1)))
gausFreq1 = fft.fft2(gausFilter1)
transGausFreq1 = fft.fftshift(gausFreq1)

sigma2 = 1
d2 = sigma2 * sigma2
gausFilter2 = utils.GausFilter2d((image.shape[0], image.shape[1]), ((d2, 0), (0, d2)))
gausFreq2 = fft.fft2(gausFilter2)
transGausFreq2 = fft.fftshift(gausFreq2)

imageFreq = fft.fft2(image, axes=[0, 1])
transImageFreq = fft.fftshift(imageFreq, axes=[0, 1])

transBlurFreq1 = transImageFreq * np.expand_dims(transGausFreq1, -1)
blurFreq1 = fft.ifftshift(transBlurFreq1, axes=[0, 1])
blur1 = fft.ifft2(blurFreq1, axes=[0, 1])
blur1 = fft.ifftshift(blur1, axes=[0, 1])

transBlurFreq2 = transImageFreq * np.expand_dims(transGausFreq2, -1)
blurFreq2 = fft.ifftshift(transBlurFreq2, axes=[0, 1])
blur2 = fft.ifft2(blurFreq2, axes=[0, 1])
blur2 = fft.ifftshift(blur2, axes=[0, 1])

plt.figure(figsize=(50, 30))

plt.subplot(351)
plt.title("original", fontsize=24)
plt.imshow(image)
plt.subplot(3, 5, 6)
plt.imshow(np.log(1 + abs(transImageFreq[:, :, 0])), cmap="gray")
plt.subplot(3, 5, 11)
plt.imshow(np.angle(transImageFreq[:, :, 0]), cmap="gray")

plt.subplot(352)
plt.title("sigma:10", fontsize=24)
plt.imshow(gausFilter1, cmap="gray")
plt.subplot(3, 5, 7)
plt.imshow(np.log(1 + abs(transGausFreq1)), cmap="gray")
plt.subplot(3, 5, 12)
plt.imshow(np.angle(transGausFreq1), cmap="gray")

plt.subplot(353)
plt.title("sigma:10", fontsize=24)
plt.imshow(abs(blur1))
plt.subplot(3, 5, 8)
plt.imshow(np.log(1 + abs(transBlurFreq1[:, :, 0])), cmap="gray")
plt.subplot(3, 5, 13)
plt.imshow(np.angle(transBlurFreq1[:, :, 0]), cmap="gray")

plt.subplot(354)
plt.title("sigma:1", fontsize=24)
plt.imshow(gausFilter2, cmap="gray")
plt.subplot(3, 5, 9)
plt.imshow(np.log(1 + abs(transGausFreq2)), cmap="gray")
plt.subplot(3, 5, 14)
plt.imshow(np.angle(transGausFreq2), cmap="gray")

plt.subplot(355)
plt.title("sigma:1", fontsize=24)
plt.imshow(abs(blur2))
plt.subplot(3, 5, 10)
plt.imshow(np.log(1 + abs(transBlurFreq2[:, :, 0])), cmap="gray")
plt.subplot(3, 5, 15)
plt.imshow(np.angle(transBlurFreq2[:, :, 0]), cmap="gray")

plt.savefig(r"image/fourier.png")
plt.show()
