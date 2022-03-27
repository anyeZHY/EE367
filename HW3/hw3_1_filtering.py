import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import skimage.io as io
from scipy.signal import convolve2d
from time import process_time
from pypher.pypher import psf2otf
import matplotlib.pyplot as plt

from fspecial import fspecial_gaussian_2d

img = io.imread('birds_gray.png').astype(float)/255

sigmas = [0.1, 1, 10]
for sigma in sigmas:
    filtSize = np.ceil(9 * sigma).astype(int)
    lp = fspecial_gaussian_2d((filtSize, filtSize), sigma)

    ### Your code here ###
