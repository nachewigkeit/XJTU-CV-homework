import itertools as iter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# TODO
def interpolate(img, pos):
    x = pos[0, :]
    y = pos[1, :]
    xLow = np.floor(x)
    xHigh = np.ceil(x)
    yLow = np.floor(y)
    yHigh = np.ceil(y)

    xLowDiff = x - xLow
    xHighDiff = xHigh - x
    yLowDiff = y - yLow
    yHighDiff = yHigh - y

    xValue = xLow
    xValue[xHighDiff < xLowDiff] = xHigh[xHighDiff < xLowDiff]
    xValue = xValue.astype('int')

    yValue = yLow
    yValue[yHighDiff < yLowDiff] = yHigh[yHighDiff < yLowDiff]
    yValue = yValue.astype('int')

    legal = (xValue >= 0) & (xValue < img.shape[0]) & (yValue >= 0) & (yValue < img.shape[1])
    value = np.zeros((pos.shape[1], img.shape[2]))
    value[legal, :] = img[xValue[legal], yValue[legal], :]
    return value


def getOldPos(shape, T):
    T = np.linalg.inv(T)
    newPos = np.ones((3, shape[0] * shape[1]))
    x, y = range(shape[0]), range(shape[1])
    newPos[[0, 1], :] = np.array(list(iter.product(x, y))).T
    return np.dot(T, newPos)


def transform(img, T):
    oldPos = getOldPos(img.shape[0:2], T)
    inter = interpolate(img, oldPos[0:2, :])
    inter = inter.reshape((img.shape[0], img.shape[1], -1))
    return inter


def translation(img, offset):
    T = np.array([[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]])
    return transform(img, T)


def rotate(img, angle):
    T = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return transform(img, T)


def euclidean(img, offset, angle):
    T = np.array([[np.cos(angle), -np.sin(angle), offset[0]], [np.sin(angle), np.cos(angle), offset[1]], [0, 0, 1]])
    return transform(img, T)


def similarity(img, offset, angle, ratio):
    T = np.array([[ratio * np.cos(angle), -ratio * np.sin(angle), offset[0]],
                  [ratio * np.sin(angle), ratio * np.cos(angle), offset[1]], [0, 0, 1]])
    return transform(img, T)


def affine(img, M):
    T = [M, np.array([0, 0, 1])]
    T = np.vstack(T)
    return transform(img, T)


def projective(img, T):
    oldPos = getOldPos(img.shape[0:2], T)
    oldPos = oldPos[0:2, :] / (oldPos[2, :] + 1e-6)
    inter = interpolate(img, oldPos)
    inter = inter.reshape((img.shape[0], img.shape[1], -1))
    return inter


if __name__ == "__main__":
    img = Image.open(r"image\cake.jpg")
    img = np.array(img) / 255

    # newimg = translation(img, (50, 100))
    # newimg = rotate(img, np.pi / 6)
    # newimg = euclidean(img, (50, 100), np.pi / 6)
    # newimg = similarity(img, (50, 100), np.pi / 6, 0.5)
    # M = np.array([[1, 0.2, 50], [0.2, 1, 100]])
    # newimg = affine(img, M)
    T = np.array([[1, 0.1, 50], [0.1, 1, 100], [0, 0.001, 1]])
    newimg = projective(img, T)
    plt.imshow(newimg)
    plt.show()
