"""
threshold the gradient magnitude image to detect the edges.
"""
import cv2
import numpy as np
from cv_cw1.paths import KITTY_BMP
from cv_cw1.task_1_convolve import convolve
from cv_cw1.task_2_compute import compute_gradients, compute_magnitude
from matplotlib import pyplot as plt


def threshold(img: np.array, thresh: int) -> np.array:
    _, output = cv2.threshold(img, thresh=thresh,
                              maxval=255, type=cv2.THRESH_BINARY)
    return output


def nothing(thresh):
    print(thresh)


def main():
    kitty: np.array = cv2.imread(KITTY_BMP, cv2.IMREAD_GRAYSCALE)
    # convolve the kitty
    kitty_w_m = convolve(kitty, mode='weighted_mean')
    # get the gradients
    kitty_w_m_grads = compute_gradients(kitty_w_m, mode='sobel')
    # get the gradient magnitude
    kitty_w_m_mag = compute_magnitude(kitty_w_m_grads[0], kitty_w_m_grads[1])

    cv2.namedWindow('edge-detection')
    cv2.createTrackbar('thresh', 'edge-detection', 0, 255, nothing)
    thresh = 0
    while True:
        cv2.imshow('edge-detection', threshold(kitty_w_m_mag, thresh))
        if cv2.waitKey(1) & 0xFF == 27:
            break
        thresh = int(cv2.getTrackbarPos('thresh', 'edge-detection'))
    cv2.destroyAllWindows()

    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(threshold(kitty_w_m_mag, 30), cmap='gray')  # bad
    axarr[1].imshow(threshold(kitty_w_m_mag, 109), cmap='gray')  # optimal
    axarr[2].imshow(threshold(kitty_w_m_mag, 240), cmap='gray')  # too much
    plt.show()


if __name__ == '__main__':
    main()
