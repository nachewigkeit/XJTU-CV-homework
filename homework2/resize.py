import numpy as np
from PIL import Image
from utils import subsampling, upsampling
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

if __name__ == "__main__":
    '''
    img = Image.open(r"image\cake.jpg")
    img = np.array(img) / 255
    mid = (400, 650)
    step = 20
    margin = 5
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    newimg = upsampling(img[mid[0] - step:mid[0] + step, mid[1] - step:mid[1] + step, :], (4, 4), "nearest")
    plt.title("nearest", fontsize=36)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(newimg[margin:-margin, margin:-margin, :])

    plt.subplot(122)
    newimg = upsampling(img[mid[0] - step:mid[0] + step, mid[1] - step:mid[1] + step, :], (4, 4), "bilinear")
    plt.title("bilinear", fontsize=36)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(newimg[margin:-margin, margin:-margin, :])

    plt.savefig("image/interpolate.png", bbox_inches='tight', pad_inches=0)
    '''

    img = Image.open(r"image\alias_origin.jpg")
    img = np.array(img) / 255
    gausImg = gaussian_filter(img, 3)
    smallImg = subsampling(img, (0.25, 0.25))
    smallGausImg = subsampling(gausImg, (0.25, 0.25))

    plt.figure(figsize=(20, 20))
    plt.subplot(221)
    plt.title("no gaus", fontsize=36)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='gray')
    plt.subplot(222)
    plt.title("sigma:3", fontsize=36)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(gausImg, cmap='gray')
    plt.subplot(223)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(smallImg, cmap='gray')
    plt.subplot(224)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(smallGausImg, cmap='gray')
    plt.savefig("image/alias.png", bbox_inches='tight', pad_inches=0)
