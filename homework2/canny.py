import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from utils import interpolate
from queue import Queue


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


def NMS(xEdge, yEdge):
    shape = xEdge.shape
    mag = np.sqrt(xEdge * xEdge + yEdge * yEdge)
    xStep = xEdge / mag
    yStep = yEdge / mag
    step = np.vstack([xStep.flatten(), yStep.flatten()])

    x, y = range(shape[0]), range(shape[1])
    oldPos = np.array(list(itertools.product(x, y))).T
    newPos1 = oldPos + step
    newPos2 = oldPos - step
    newmag1 = interpolate(mag, newPos1).reshape((shape[0], shape[1]))
    newmag2 = interpolate(mag, newPos2).reshape((shape[0], shape[1]))

    mag[(mag < newmag1) | (mag < newmag2)] = 0
    return mag


def biThres(mag, low, high):
    pos = np.argwhere(mag > high)
    visited = np.zeros((mag.shape[0], mag.shape[1]))
    waiting = Queue()

    for i in range(pos.shape[0]):
        waiting.put(list(pos[i, :]))

    while not waiting.empty():
        now = waiting.get()
        x, y = now
        visited[x, y] = 1

        if x > 0 and y > 0 and mag[x - 1, y - 1] > low:
            new = [x - 1, y - 1]
            if visited[new[0], new[1]] == 0:
                waiting.put(new)
        if x > 0 and y < mag.shape[1] - 1 and mag[x - 1, y + 1] > low:
            new = [x - 1, y + 1]
            if visited[new[0], new[1]] == 0:
                waiting.put(new)
        if x < mag.shape[0] - 1 and y > 0 and mag[x + 1, y - 1] > low:
            new = [x + 1, y - 1]
            if visited[new[0], new[1]] == 0:
                waiting.put(new)
        if x < mag.shape[0] - 1 and y < mag.shape[1] - 1 and mag[x + 1, y + 1] > low:
            new = [x + 1, y + 1]
            if visited[new[0], new[1]] == 0:
                waiting.put(new)

    return visited


if __name__ == "__main__":
    img = Image.open(r"image\cake.jpg")
    img = img.convert('L')
    img = np.array(img, dtype='float32') / 255

    sigma = 5
    xkernel = gausDerivativeFilter(sigma, 0)
    xEdge = cv.filter2D(img, -1, xkernel)

    ykernel = gausDerivativeFilter(sigma, 1)
    yEdge = cv.filter2D(img, -1, ykernel)

    mag = np.sqrt(xEdge * xEdge + yEdge * yEdge)
    angle = np.arctan(yEdge / xEdge)

    '''
    plt.imshow(xkernel, cmap='bwr')
    plt.show()
    plt.imshow(np.sqrt(mag), cmap='gray')
    plt.show()
    plt.imshow(angle, cmap='bwr')
    plt.show()
    '''

    nms = NMS(xEdge, yEdge)
    nms /= nms.max()

    plt.imshow(nms, cmap='gray')
    plt.show()

    edge = biThres(nms, 0.1, 0.2)
    plt.imshow(edge, cmap='gray')
    plt.show()
