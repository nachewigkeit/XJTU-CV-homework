from scipy.ndimage import sobel, gaussian_filter
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageChops
import matplotlib.pyplot as plt


def NMS(corner, thres):
    answer = []
    for i in range(1, corner.shape[0] - 1):
        for j in range(1, corner.shape[1] - 1):
            if corner[i, j] > thres and corner[i, j] == corner[i - 1:i + 2, j - 1:j + 2].max():
                answer.append([i, j])

    answer = np.array(answer)
    return answer


def harris(colorImg, sigma, thres):
    img = colorImg.convert('L')
    img = np.array(img, dtype='float32') / 255

    xEdge = sobel(img, 0)
    yEdge = sobel(img, 1)

    xx = gaussian_filter(xEdge * xEdge, sigma)
    xy = gaussian_filter(xEdge * yEdge, sigma)
    yy = gaussian_filter(yEdge * yEdge, sigma)

    det = xx * yy - xy * xy
    tr = xx + yy
    corner = det / (tr + 1e-6)
    corner /= corner.max()
    answer = NMS(corner, thres)

    return corner, answer


colorImg = Image.open(r"image\cake.jpg")

'''
sigma = [1, 5, 10]
corners = []
answers = []

length = len(sigma)
plt.figure(figsize=(10 * length, 15))
for i in range(length):
    plt.subplot(2, length, i + 1)
    plt.title("sigma:" + str(sigma[i]), fontsize=36)
    plt.xticks([])
    plt.yticks([])
    corner, answer = harris(colorImg, sigma[i], 0.3)
    plt.imshow(corner, cmap='gray')

    plt.subplot(2, length, length + i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.array(colorImg))
    plt.scatter(answer[:, 1], answer[:, 0], s=50, marker='x')
plt.savefig("image/corner.png", bbox_inches='tight', pad_inches=0)
'''

images = []

bright = ImageEnhance.Brightness(colorImg)
brightImg = bright.enhance(1.2)
images.append(brightImg)
brightImg = bright.enhance(0.8)
images.append(brightImg)
contrast = ImageEnhance.Contrast(colorImg)
contrastImg = contrast.enhance(1.2)
images.append(contrastImg)
contrastImg = contrast.enhance(0.8)
images.append(contrastImg)

rotate = colorImg.rotate(45)
images.append(rotate)
trans = ImageChops.offset(colorImg, 100, 200)
trans.paste((0, 0, 0), (0, 0, 100, colorImg.size[1]))
trans.paste((0, 0, 0), (0, 0, colorImg.size[0], 200))
images.append(trans)
small = colorImg.resize((int(colorImg.size[0] * 0.25), int(colorImg.size[1] * 0.25)))
images.append(small)
big = colorImg.resize((int(colorImg.size[0] * 4), int(colorImg.size[1] * 4)))
images.append(big)

title = ["High Brightness", "Low Brightness", "High Contrast", "Low Contrast",
         "rotate", "translation", "0.25x", "4x"]
plt.figure(figsize=(40, 15))
for i in range(len(images)):
    plt.subplot(2, 4, i + 1)
    plt.title(title[i], fontsize=36)
    plt.xticks([])
    plt.yticks([])
    corner, answer = harris(images[i], 5, 0.3)
    plt.imshow(np.array(images[i]))
    plt.scatter(answer[:, 1], answer[:, 0], s=50, marker='x')
plt.savefig("image/variance.png", bbox_inches='tight', pad_inches=0)
