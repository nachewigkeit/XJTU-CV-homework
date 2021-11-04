import numpy as np
from PIL import Image
from resize import subsampling, upsampling
from utils import imgSave


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
    step = 512
    start = (394, 175)
    img = img.crop((start[0], start[1], start[0] + step, start[1] + step))
    img = np.array(img, dtype='float32').reshape((step, step, 1))

    layer = 5

    pyramid = laplacePyramid(img, layer)
    for i in range(len(pyramid)):
        imgSave(pyramid[i].squeeze(), "laplace" + str(i), 'gray')

    pyramid = gausPyramid(img, layer)
    for i in range(len(pyramid)):
        imgSave(pyramid[i].squeeze(), "gaus" + str(i), 'gray')
