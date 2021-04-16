import cv2
import numpy as np
from cv_cw1.paths import KITTY_BMP
from matplotlib import pyplot as plt
# the kernels
MEAN_KERNEL = np.ones(shape=(3, 3)) / 10  # not weighted, normalised. (add up to 9)
WEIGHTED_MEAN_KERNEL = np.array([[0.5, 1, 0.5],
                                 [1,   2,   1],
                                 [0.5, 1, 0.5]]) / 8  # weighted, normalised (add up to 8)

gau_1d = cv2.getGaussianKernel(3, sigma=1)
GAUSSIAN_KERNEL = np.outer(gau_1d, gau_1d.transpose())


def pad(img: np.array) -> np.array:
    """
    :param img: an image to pad.
    :return:
    """
    # 1. build a placeholder.
    img_padded = np.zeros(shape=(img.shape[0] + 2, img.shape[1] + 2))
    # 2. copy and paste inside the "box"
    img_padded[1:-1, 1:-1] = img
    return img_padded


def convolve(img: np.array, mode: str = 'mean') -> np.array:
    """
     :param img:
     :param mode:
     :return:
     """
    # choose a kernel accordingly
    global MEAN_KERNEL, WEIGHTED_MEAN_KERNEL
    if mode == "mean":
        kernel = MEAN_KERNEL
    elif mode == "weighted_mean":
        kernel = WEIGHTED_MEAN_KERNEL
    elif mode == "gaussian":
        kernel = GAUSSIAN_KERNEL
    else:
        raise ValueError
    # pad the image
    img_padded = pad(img)
    # each dimension should have incremented by a value of 2.
    assert img_padded.shape == (img.shape[0] + 2, img.shape[1] + 2)
    # init an image to store the convoluted values.
    img_convoluted = np.zeros((img.shape[0], img.shape[1]))
    # loop over the dimensions of the original image
    for x_idx in range(img_convoluted.shape[0]):  #
        for y_idx in range(img_convoluted.shape[1]):
            # component-wise multiply, and them sum them up
            # here, we don't need division as the weights of the kernel is normalised.
            img_convoluted[x_idx, y_idx] = (kernel * img_padded[x_idx: x_idx + 3, y_idx: y_idx + 3]).sum()
    return img_convoluted


def main():
    # load kitty image in gray scale
    kitty: np.array = cv2.imread(KITTY_BMP, cv2.IMREAD_GRAYSCALE)
    # convolve the image.
    kitty_conv_mean = convolve(kitty, mode='mean')
    kitty_conv_w_mean = convolve(kitty, mode='weighted_mean')
    # make sure that the shape did not.
    assert kitty.shape == kitty_conv_mean.shape
    assert kitty.shape == kitty_conv_w_mean.shape
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(kitty, cmap='gray')
    axarr[1].imshow(kitty_conv_mean, cmap='gray')
    axarr[2].imshow(kitty_conv_w_mean, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
