"""
task_2 - compute the horizontal & vertical gradients of the image.
"""
from typing import Tuple
import cv2
import numpy as np
from cv_cw1.paths import KITTY_BMP
from cv_cw1.task_1_convolve import pad, convolve
from cv_cw1.utils import show_img

# convolving an image with prewitt kernels results in an approximation of
# the differentiation of the image with respect to the pixel values.
PREWITT_KERNELS = (
    np.array([[-1, 0, 1],
              [-1, 0, 1],
              [-1, 0, 1]]),
    np.array([[-1, -1, -1],
              [0,   0,  0],
              [1,   1,  1]])
)
# convolving an image with SOBEL_KERNEL also approximates the differentiation,
# but the gradients are weighted towards the center.
SOBEL_KERNELS = (
    np.array([[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]),
    np.array([[-1, -2, -1],
              [0,   0,  0],
              [1,   2,  1]])
)


def compute_gradients(img: np.array, mode: str) -> Tuple[np.array, np.array]:
    global PREWITT_KERNELS, SOBEL_KERNELS
    if mode == "prewitt":
        kernels = PREWITT_KERNELS
    elif mode == "sobel":
        kernels = SOBEL_KERNELS
    else:
        raise ValueError
    # pad the image
    img_padded = pad(img)
    # init a placeholder for the gradients. (x - horizontal, y - vertical)
    img_x_gradients = np.zeros(shape=img.shape)
    img_y_gradients = np.zeros(shape=img.shape)
    for y_idx in range(img_x_gradients.shape[1]):
        for x_idx in range(img_x_gradients.shape[0]):  # horizontal convolution
            img_x_gradients[x_idx, y_idx] = (kernels[0] * img_padded[x_idx: x_idx + 3, y_idx: y_idx + 3]).sum()
    for x_idx in range(img_x_gradients.shape[0]):
        for y_idx in range(img_x_gradients.shape[1]):  # vertical convolution
            img_y_gradients[x_idx, y_idx] = (kernels[1] * img_padded[x_idx: x_idx + 3, y_idx: y_idx + 3]).sum()
    return img_x_gradients, img_y_gradients


def compute_magnitude(x_gradients: np.array, y_gradients: np.array) -> np.array:
    """
    sqrt(x_g**2 + y_g**2)
    :param x_gradients:
    :param y_gradients:
    :return:
    """
    return np.sqrt(x_gradients**2 , y_gradients**2)


def main():
    # load kitty image in gray scale
    kitty: np.array = cv2.imread(KITTY_BMP, cv2.IMREAD_GRAYSCALE)
    # convolve the image
    kitty_convoluted = convolve(kitty, mode="weighted_mean")
    # compute the horizontal & vertical gradients.
    prewitt_gradients = compute_gradients(kitty_convoluted, 'prewitt')
    sobel_gradients = compute_gradients(kitty_convoluted, 'sobel')
    prewitt_magnitude = compute_magnitude(prewitt_gradients[0], prewitt_gradients[1])
    sobel_magnitude = compute_magnitude(sobel_gradients[0], sobel_gradients[1])
    show_img(prewitt_gradients[0])
    show_img(prewitt_gradients[1])
    show_img(sobel_gradients[0])
    show_img(sobel_gradients[0])
    show_img(prewitt_magnitude)
    show_img(sobel_magnitude)


if __name__ == '__main__':
    main()
