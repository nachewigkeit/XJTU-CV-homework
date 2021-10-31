import numpy as np
from PIL import Image
from resize import subsampling, upsampling
import matplotlib.pyplot as plt


def gausPyramid(img, layer):
    answer = []
    newimg = img
    for i in range(layer):
        answer.append(newimg)
        newimg = subsampling(newimg, (0.5, 0.5))

    return answer


def laplacePyramid(img, layer):
    gaus = gausPyramid(img, layer)
    gaus.reverse()
    answer = [gaus[0]]
    for i in range(1, layer):
        newimg = gaus[i] - upsampling(gaus[i - 1], (2, 2), "nearest")
        answer.append(newimg)

    return answer


if __name__ == "__main__":
    img = Image.open(r"image\cake.jpg")
    img = img.convert('L')
    start = (122, 59)
    img = img.crop((start[0], start[1], start[0] + 256, start[1] + 256))
    img = np.array(img).reshape((256, 256, 1))

    layer = 5
    pyramid = laplacePyramid(img, layer)

    plt.figure(figsize=(10 * layer, 10))
    for i in range(len(pyramid)):
        plt.subplot(1, layer, i+1)
        plt.imshow(pyramid[i], cmap='gray')
    plt.show()