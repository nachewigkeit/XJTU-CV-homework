import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import interpolate, getGrad
from queue import Queue


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

    xEdge, yEdge = getGrad(img, 3)
    mag = np.sqrt(xEdge * xEdge + yEdge * yEdge)
    angle = np.arctan(yEdge / xEdge)

    nms = NMS(xEdge, yEdge)
    nms /= nms.max()

    edge = biThres(nms, 0.1, 0.2)

    mid = (400, 650)
    step = 200
    plt.figure(figsize=(30, 10))
    plt.subplot(131)
    plt.title("gradient", fontsize=36)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(mag[mid[0] - step:mid[0] + step, mid[1] - step:mid[1] + step], cmap='gray')
    plt.subplot(132)
    plt.title("nms", fontsize=36)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(nms[mid[0] - step:mid[0] + step, mid[1] - step:mid[1] + step], cmap='gray')
    plt.subplot(133)
    plt.title("link", fontsize=36)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(edge[mid[0] - step:mid[0] + step, mid[1] - step:mid[1] + step], cmap='gray')
    plt.savefig("image/canny.png", bbox_inches='tight', pad_inches=0)
