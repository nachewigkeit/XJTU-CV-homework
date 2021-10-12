import numpy as np
from scipy.stats import multivariate_normal, norm


def GausFilter1d(size, sigma):
    distribution = norm(0, sigma)
    pos = np.zeros(size)
    for i in range(size):
        pos[i] = i - (size - 1) / 2
    prob = distribution.pdf(pos)
    prob /= np.sum(prob)
    return prob


def GausFilter2d(size, sigma):
    distribution = multivariate_normal((0, 0), sigma)
    pos = np.zeros((size[0], size[1], 2))
    for i in range(size[0]):
        for j in range(size[1]):
            pos[i, j, :] = (i - (size[0] - 1) / 2, j - (size[1] - 1) / 2)
    prob = distribution.pdf(pos)
    prob /= np.sum(prob)
    return prob


def padding(img, method, border):
    if len(img.shape) == 2:
        img = img.reshape((img.shape[0], img.shape[1], 1))

    shape = img.shape
    imgPadArray = np.zeros((shape[0] + border[1] + border[3], shape[1] + border[0] + border[2], shape[2]))
    if border[2] == 0 and border[3] == 0:
        imgPadArray[border[1]:, border[0]:] = img
    elif border[2] == 0 and border[3] != 0:
        imgPadArray[border[1]:-border[3], border[0]:] = img
    elif border[2] != 0 and border[3] == 0:
        imgPadArray[border[1]:, border[0]:-border[2]] = img
    else:
        imgPadArray[border[1]:-border[3], border[0]:-border[2]] = img

    if method == "wrap":
        # edge
        imgPadArray[0:border[1], border[0]:-border[2]] = img[-border[1]:, :]
        imgPadArray[-border[3]:, border[0]:-border[2]] = img[0:border[3], :]
        imgPadArray[border[1]:-border[3], 0:border[0]] = img[:, -border[0]:]
        imgPadArray[border[1]:-border[3], -border[2]:] = img[:, 0:border[2]]

        # corner
        imgPadArray[0:border[1], 0:border[0]] = img[-border[1]:, -border[0]:]
        imgPadArray[0:border[1], -border[2]:] = img[-border[1]:, 0:border[2]]
        imgPadArray[-border[3]:, 0:border[0]] = img[0:border[3], -border[0]:]
        imgPadArray[-border[3]:, -border[2]:] = img[0:border[3], 0:border[2]]
    elif method == "reflect":
        # edge
        imgPadArray[0:border[1], border[0]:-border[2]] = np.flip(img[0:border[1], :], 0)
        imgPadArray[-border[3]:, border[0]:-border[2]] = np.flip(img[-border[3]:, :], 0)
        imgPadArray[border[1]:-border[3], 0:border[0]] = np.flip(img[:, 0:border[0]], 1)
        imgPadArray[border[1]:-border[3], -border[2]:] = np.flip(img[:, -border[2]:], 1)

        # corner
        imgPadArray[0:border[1], 0:border[0]] = np.flip(img[0:border[1], 0:border[0]], [0, 1])
        imgPadArray[0:border[1], -border[2]:] = np.flip(img[0:border[1], -border[2]:], [0, 1])
        imgPadArray[-border[3]:, 0:border[0]] = np.flip(img[-border[3]:, 0:border[0]], [0, 1])
        imgPadArray[-border[3]:, -border[2]:] = np.flip(img[-border[3]:, -border[2]:], [0, 1])
    elif method == "copy":
        # edge
        imgPadArray[0:border[1], border[0]:-border[2]] = img[[0], :]
        imgPadArray[-border[3]:, border[0]:-border[2]] = img[[-1], :]
        imgPadArray[border[1]:-border[3], 0:border[0]] = img[:, [0]]
        imgPadArray[border[1]:-border[3], -border[2]:] = img[:, [-1]]

        # corner
        imgPadArray[0:border[1], 0:border[0]] = img[0, 0]
        imgPadArray[0:border[1], -border[2]:] = img[0, -1]
        imgPadArray[-border[3]:, 0:border[0]] = img[-1, 0]
        imgPadArray[-border[3]:, -border[2]:] = img[-1, -1]

    return imgPadArray


def conv(img, filter, method, border):
    imgPad = padding(img, method, border)
    imgConv = np.zeros((imgPad.shape[0] - filter.shape[0] + 1, imgPad.shape[1] - filter.shape[1] + 1, imgPad.shape[2]))
    for k in range(imgConv.shape[2]):
        for i in range(imgConv.shape[0]):
            for j in range(imgConv.shape[1]):
                imgConv[i, j, k] = np.sum(filter * imgPad[i:i + filter.shape[0], j:j + filter.shape[1], k])

    return imgConv


def bilateralConv(img, filter, sigma, method, border):
    imgPad = padding(img, method, border)
    imgConv = np.zeros((imgPad.shape[0] - filter.shape[0] + 1, imgPad.shape[1] - filter.shape[1] + 1, imgPad.shape[2]))
    distribution = norm(0, sigma)
    for k in range(imgConv.shape[2]):
        for i in range(imgConv.shape[0]):
            for j in range(imgConv.shape[1]):
                colorFilter = imgPad[i:i + filter.shape[0], j:j + filter.shape[1], k] - \
                              imgPad[i + (filter.shape[0] - 1) // 2, j + (filter.shape[1] - 1) // 2, k]
                prob = distribution.pdf(colorFilter)
                colorFilter = prob * filter
                colorFilter = colorFilter / (colorFilter.sum() + 1e-6)
                imgConv[i, j, k] = np.sum(colorFilter * imgPad[i:i + filter.shape[0], j:j + filter.shape[1], k])

    return imgConv


if __name__ == "__main__":
    print(GausFilter1d(3, 1))
    print(GausFilter1d(5, 1))
