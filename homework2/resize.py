import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import subsampling, upsampling

if __name__ == "__main__":
    img = Image.open(r"image\cake.jpg")
    img = np.array(img) / 255
    # newimg = subsampling(img, (0.25, 0.25))
    newimg = upsampling(img, (4, 4), "nearest")
    plt.imshow(newimg)
    plt.show()
