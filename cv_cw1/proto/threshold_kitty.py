"""
from snippet 9.
Threshold a greyscale image.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from cv_cw1.paths import KITTY_BMP


def main():
    # read kitty as a gray scale image
    kitty: np.array = cv2.imread(KITTY_BMP, cv2.IMREAD_GRAYSCALE)
    plt.imshow(kitty)
    plt.show()
    # Threshold manually at intensity level 150
    # Note that threshold() returns the computed threshold value
    # and the resulting image.  We don't need the value so we put
    # it in _.
    _, output = cv2.threshold(kitty, thresh=150,
                              maxval=255, type=cv2.THRESH_BINARY)
    plt.imshow(output)
    plt.show()


if __name__ == '__main__':
    main()
