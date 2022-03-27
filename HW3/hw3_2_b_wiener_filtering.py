import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from pypher.pypher import psf2otf

from pdb import set_trace
from fspecial import fspecial_gaussian_2d

img = io.imread('birds_gray.png').astype(float)/255

# Task 2b - Wiener filtering

c = fspecial_gaussian_2d((35, 35), 5.)
cFT = psf2otf(c, img.shape)
# Blur image with kernel
blur = np.zeros_like(img)

sigmas = [0, 0.001, 0.01, 0.1]
for sigma in sigmas:
    # Add noise to blurred image
    unfilt = blur + sigma * np.random.randn(*blur.shape)

    ### Your code here ###
