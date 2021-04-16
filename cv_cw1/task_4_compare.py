"""
compare the result of using mean kernel with weighted mean kernel.
"""
import cv2
import numpy as np
from cv_cw1.paths import KITTY_BMP
from cv_cw1.task_1_convolve import convolve
from cv_cw1.task_2_compute import compute_gradients, compute_magnitude
from cv_cw1.task_3_threshold import threshold
from cv_cw1.utils import show_img


def main():
    kitty: np.array = cv2.imread(KITTY_BMP, cv2.IMREAD_GRAYSCALE)
    kitty_conv_m = convolve(kitty, mode='mean')
    kitty_conv_w_m = convolve(kitty, mode='weighted_mean')
    kitty_m_grads = compute_gradients(kitty_conv_m, mode='sobel')
    kitty_w_m_grads = compute_gradients(kitty_conv_w_m, mode='sobel')
    kitty_m_mag = compute_magnitude(kitty_m_grads[0], kitty_m_grads[1])
    kitty_w_m_mag = compute_magnitude(kitty_w_m_grads[0], kitty_w_m_grads[1])
    kitty_m_edges = threshold(kitty_m_mag)
    kitty_w_m_edges = threshold(kitty_w_m_mag)
    # notice the difference!
    show_img(kitty_m_edges)
    show_img(kitty_w_m_edges)


if __name__ == '__main__':
    main()
