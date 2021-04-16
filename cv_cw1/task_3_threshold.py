"""
threshold the gradient magnitude image to detect the edges.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from cv_cw1.paths import KITTY_BMP
from cv_cw1.task_1_convolve import convolve
from cv_cw1.task_2_compute import compute_gradients, compute_magnitude
from cv_cw1.utils import show_img


def plot_hist(img: np.array):
    # Calculate the histogram
    hist = cv2.calcHist(images=[img],
                        channels=[0],
                        mask=None,
                        histSize=[256],
                        ranges=[0, 256])
    hist = hist.reshape(256)
    # Plot histogram
    plt.bar(np.linspace(0, 255, 256), hist)
    plt.title('Histogram')
    plt.ylabel('Frequency')
    plt.xlabel('Grey Level')
    plt.show()


def threshold(img: np.array, val: int) -> np.array:
    _, output = cv2.threshold(img, thresh=val,
                              maxval=255, type=cv2.THRESH_BINARY)
    return output


def main():
    kitty: np.array = cv2.imread(KITTY_BMP, cv2.IMREAD_GRAYSCALE)
    # convolve the kitty
    kitty_convoluted = convolve(kitty, mode='weighted_mean')
    # get the gradients
    kitty_gradients = compute_gradients(kitty_convoluted, mode='sobel')
    # get the gradient magnitude
    kitty_magnitude = compute_magnitude(kitty_gradients[0], kitty_gradients[1])
    kitty_edges = threshold(kitty_magnitude, 120)
    # have a look at the histogram
    plot_hist(kitty)
    plot_hist(np.float32(kitty_convoluted))
    plot_hist(np.float32(kitty_magnitude))
    show_img(kitty_edges)


if __name__ == '__main__':
    main()
