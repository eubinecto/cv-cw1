import numpy as np
from matplotlib import pyplot as plt


def show_img(img: np.array):
    plt.imshow(img, cmap='gray')
    plt.show()
