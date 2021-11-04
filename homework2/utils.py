import numpy as np
import itertools
import matplotlib.pyplot as plt
import cv2 as cv


def interpolate(img, pos, method="bilinear"):
    img = img.reshape((img.shape[0], img.shape[1], -1))

    x = pos[0, :]
    y = pos[1, :]
    xLow = np.floor(x).astype('int')
    xHigh = np.ceil(x).astype('int')
    xHigh[xHigh == xLow] += 1
    yLow = np.floor(y).astype('int')
    yHigh = np.ceil(y).astype('int')
    yHigh[yHigh == yLow] += 1

    xLowDiff = x - xLow
    xHighDiff = xHigh - x
    yLowDiff = y - yLow
    yHighDiff = yHigh - y

    value = np.zeros((pos.shape[1], img.shape[2]))
    if method == "nearest":
        xValue = xLow
        xValue[xHighDiff < xLowDiff] = xHigh[xHighDiff < xLowDiff]
        xValue = xValue

        yValue = yLow
        yValue[yHighDiff < yLowDiff] = yHigh[yHighDiff < yLowDiff]
        yValue = yValue

        legal = (xValue >= 0) & (xValue < img.shape[0]) & (yValue >= 0) & (yValue < img.shape[1])
        value[legal, :] = img[xValue[legal], yValue[legal], :]
    else:
        legal = (xLow >= 0) & (xHigh < img.shape[0]) & (yLow >= 0) & (yHigh < img.shape[1])

        xLyL = (xHighDiff * yHighDiff)[legal].reshape((-1, 1))
        xLyH = (xHighDiff * yLowDiff)[legal].reshape((-1, 1))
        xHyL = (xLowDiff * yHighDiff)[legal].reshape((-1, 1))
        xHyH = (xLowDiff * yLowDiff)[legal].reshape((-1, 1))

        xLowLegal = xLow[legal]
        xHighLegal = xHigh[legal]
        yLowLegal = yLow[legal]
        yHighLegal = yHigh[legal]

        value[legal, :] = xLyL * img[xLowLegal, yLowLegal, :] + xLyH * img[xLowLegal, yHighLegal, :] + \
                          xHyL * img[xHighLegal, yLowLegal, :] + xHyH * img[xHighLegal, yHighLegal, :]

    return value


def subsampling(img, rate, method="area"):
    oldshape = img.shape
    newshape = (np.ceil(oldshape[0] * rate[0]).astype('int'), np.ceil(oldshape[1] * rate[1]).astype('int'), oldshape[2])
    newimg = np.zeros(newshape)

    if method == "area":
        for i in range(newshape[0]):
            xstart = np.ceil(i / rate[0]).astype('int')
            xend = np.ceil((i + 1) / rate[0]).astype('int')
            for j in range(newshape[1]):
                ystart = np.ceil(j / rate[0]).astype('int')
                yend = np.ceil((j + 1) / rate[0]).astype('int')

                sub = img[xstart:xend, ystart:yend, :]
                newimg[i, j] = np.mean(sub, axis=(0, 1))
    else:
        for i in range(newshape[0]):
            for j in range(newshape[1]):
                newimg[i, j] = img[int(np.floor(i / rate[0])), int(np.floor(j / rate[1]))]

    return newimg


def upsampling(img, rate, method="bilinear"):
    oldshape = img.shape
    newshape = (np.ceil(oldshape[0] * rate[0]).astype('int'), np.ceil(oldshape[1] * rate[1]).astype('int'))

    x, y = np.array(range(newshape[0])) / rate[0], np.array(range(newshape[1])) / rate[1]
    pos = np.array(list(itertools.product(x, y))).T

    inter = interpolate(img, pos, method)
    inter = inter.reshape((newshape[0], newshape[1], -1))
    return inter


def imgSave(img, name, cmap=None):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap)
    plt.savefig("image/" + name + ".png", bbox_inches='tight', pad_inches=0)


def gausDerivative(sigma, pos, direct):
    x, y = pos
    return -1 / (2 * np.pi * (sigma ** 4)) * pos[direct] * np.exp(-(x * x + y * y) / (2 * sigma * sigma))


def gausDerivativeFilter(sigma, direct):
    gausSize = 25
    filter = np.zeros((gausSize, gausSize))
    mid = (gausSize - 1) // 2
    for i in range(gausSize):
        for j in range(gausSize):
            filter[i, j] = gausDerivative(sigma, (i - mid, j - mid), direct)

    return filter


def getGrad(img, sigma):
    xkernel = gausDerivativeFilter(sigma, 0)
    xEdge = cv.filter2D(img, -1, xkernel)

    ykernel = gausDerivativeFilter(sigma, 1)
    yEdge = cv.filter2D(img, -1, ykernel)

    return xEdge, yEdge
