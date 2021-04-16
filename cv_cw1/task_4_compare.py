"""
compare the result of using mean kernel with weighted mean kernel.
What we should compare is the optimal value for threshold of each filter.
"""
import cv2
import numpy as np
from cv_cw1.paths import KITTY_BMP
from cv_cw1.task_1_convolve import convolve
from cv_cw1.task_2_compute import compute_gradients, compute_magnitude
from cv_cw1.task_3_threshold import threshold
from matplotlib import pyplot as plt
# the optimal values for the kernels
THRESH_M = 88
THRESH_W_M = 109


def plot_hist(img: np.array, title: str):
    # Calculate the histogram
    hist = cv2.calcHist(images=[img],
                        channels=[0],
                        mask=None,
                        histSize=[256],
                        ranges=[0, 256])
    hist = hist.reshape(256)
    # Plot histogram
    plt.bar(np.linspace(0, 255, 256), hist)
    plt.ylim(0, 1300)
    plt.title(title)
    plt.ylabel('Frequency')
    plt.xlabel('Grey Level')
    plt.show()


def main():
    kitty: np.array = cv2.imread(KITTY_BMP, cv2.IMREAD_GRAYSCALE)
    kitty_conv_m = convolve(kitty, mode='mean')
    kitty_conv_w_m = convolve(kitty, mode='weighted_mean')
    kitty_m_grads = compute_gradients(kitty_conv_m, mode='sobel')
    kitty_w_m_grads = compute_gradients(kitty_conv_w_m, mode='sobel')
    kitty_m_mag = compute_magnitude(kitty_m_grads[0], kitty_m_grads[1])
    kitty_w_m_mag = compute_magnitude(kitty_w_m_grads[0], kitty_w_m_grads[1])
    # compare the magnitudes
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(kitty_m_mag, cmap='gray')  # bad
    axarr[1].imshow(kitty_w_m_mag, cmap='gray')  # too much
    plt.show()

    # compare the histograms
    plot_hist(np.float32(kitty_m_mag), "histogram - mean kernel")
    plot_hist(np.float32(kitty_w_m_mag), "histogram - weighted mean kernel")

    # compare the edges, with the same threshold.
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(threshold(kitty_m_mag, THRESH_W_M), cmap='gray')  # for comparison
    axarr[1].imshow(threshold(kitty_w_m_mag, THRESH_W_M), cmap='gray')  # as best as it gets for w_m
    plt.show()


if __name__ == '__main__':
    main()
