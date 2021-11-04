import numpy as np
from PIL import Image
from utils import subsampling, upsampling
import matplotlib.pyplot as plt

if __name__ == "__main__":
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