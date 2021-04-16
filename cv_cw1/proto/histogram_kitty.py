"""
from snippet 7.
Create and display an image histogram using matplotlib.
"""
import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np
from cv_cw1.paths import KITTY_BMP


def main():
    img: np.array = cv2.imread(KITTY_BMP, cv2.IMREAD_GRAYSCALE)
    # Check for success
    if img is None:
        print('Failed to open', KITTY_BMP)
        sys.exit()

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


if __name__ == '__main__':
    main()
