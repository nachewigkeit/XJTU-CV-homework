from utils import gausDerivativeFilter, getGrad
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

sigma = [1, 3, 5]

length = len(sigma)
plt.figure(figsize=(10 * length, 20))
img = Image.open(r"image\cake.jpg")
img = img.convert('L')
img = np.array(img, dtype='float32') / 255
for i in range(length):
    plt.subplot(3, length, 1 + i)
    plt.xticks([])
    plt.yticks([])
    plt.title("sigma:" + str(sigma[i]), fontsize=36)
    plt.imshow(gausDerivativeFilter(sigma[i], 0), cmap='bwr')

    xEdge, yEdge = getGrad(img, sigma[i])
    mag = np.sqrt(xEdge * xEdge + yEdge * yEdge)
    angle = np.arctan(yEdge / xEdge)
    plt.subplot(3, length, length + 1 + i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.sqrt(mag), cmap='gray')
    plt.subplot(3, length, 2 * length + 1 + i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(angle, cmap='bwr')

plt.savefig("image/gausDerivative.png", bbox_inches='tight', pad_inches=0)
