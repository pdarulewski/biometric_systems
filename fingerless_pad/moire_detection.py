import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fingerless_pad.settings import IMAGES_DIR

from scipy.signal import find_peaks

# %%

def show_image():
    positive = os.path.join(IMAGES_DIR, 'positive', '10.png')
    img = cv2.imread(positive, 0)
    img = cv2.equalizeHist(img)
    img_dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(img_dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


# %%
def analyse(filename, case):
    positive = os.path.join(IMAGES_DIR, case, '{}.png'.format(filename))
    img = cv2.imread(positive, 0)
    img = cv2.equalizeHist(img)
    img_dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(img_dft)

    sigma = 0.6
    k = 2

    real = cv2.GaussianBlur(dft_shift[:, :, 0], (5, 5), sigma)
    imaginary = cv2.GaussianBlur(dft_shift[:, :, 1], (5, 5), sigma)

    magnitude_spectrum_1 = 20 * np.log(cv2.magnitude(real, imaginary))

    real = cv2.GaussianBlur(dft_shift[:, :, 0], (5, 5), k * sigma)
    imaginary = cv2.GaussianBlur(dft_shift[:, :, 1], (5, 5), k * sigma)
    magnitude_spectrum_2 = 20 * np.log(cv2.magnitude(real, imaginary))

    magnitude_spectrum = magnitude_spectrum_1 - magnitude_spectrum_2

    magnitude_spectrum = cv2.adaptiveThreshold(
        magnitude_spectrum, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()



def analyse_all():
    for filename in ['1', '2', '3', '4', '5']:
        analyse(filename, 'negative')

    for filename in ['1', '2', '3', '4', '5']:
        analyse(filename, 'positive')

# %%

def find_moire(case, filename):
    positive = os.path.join(IMAGES_DIR, case, '{}.png'.format(filename))
    img = cv2.imread(positive, cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)
    img_dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(img_dft)

    sigma = 0.5
    k = 3

    real = cv2.GaussianBlur(dft_shift[:, :, 0], (5, 5), sigma)
    imaginary = cv2.GaussianBlur(dft_shift[:, :, 1], (5, 5), sigma)

    magnitude_spectrum_1 = 20 * np.log(cv2.magnitude(real, imaginary))

    real = cv2.GaussianBlur(dft_shift[:, :, 0], (5, 5), k * sigma)
    imaginary = cv2.GaussianBlur(dft_shift[:, :, 1], (5, 5), k * sigma)
    magnitude_spectrum_2 = 20 * np.log(cv2.magnitude(real, imaginary))

    magnitude_spectrum = magnitude_spectrum_1 - magnitude_spectrum_2

    # plt.subplot(121), plt.imshow(img)
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(magnitude_spectrum)
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.show()

    upper_bound = np.percentile(magnitude_spectrum, 99)

    min_max = pd.DataFrame(magnitude_spectrum)

    min_max = min_max.where(min_max >= upper_bound, 0)
    min_max = min_max.where((min_max == 0), 255)
    min_max = min_max.values

    # plt.imshow(min_max)
    # plt.show()

    out = np.array(min_max)

    for _ in range(3):
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.morphologyEx(out, cv2.MORPH_CLOSE, se1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

        out = out * mask

    # plt.imshow(out)
    # plt.show()

    peaks, _ = find_peaks(out.ravel(), height=255, distance=100000)
    peaks_2d = np.zeros(out.ravel().shape)

    for _, value in enumerate(peaks):
        peaks_2d[value] = 1

    peaks_2d = peaks_2d.reshape(min_max.shape)

    coordinates = zip(*np.nonzero(peaks_2d))

    coordinates = list(coordinates)
    print(len(coordinates))

    output = np.array(magnitude_spectrum)

    for pair in coordinates:
        cv2.circle(output, (pair[1], pair[0]), 50, (255, 255, 255), 10)

    plt.subplot(121), plt.imshow(img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(output)
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


find_moire('positive', 10)