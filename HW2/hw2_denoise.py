from pathlib import Path
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.filters import gaussian
from scipy.ndimage import median_filter
from scipy.signal import convolve2d
from pdb import set_trace

from bilateral import bilateral2d
from fspecial import fspecial_gaussian_2d
from nlm import nonlocalmeans

clean = io.imread('night.png').astype(float)/255
noisy = io.imread('night_downsampled_noisy_sigma_0.0781.png').astype(float)/255

# Store outputs in dictionary
filtered = {}

# Choose standard deviations:
sigmas = [1, 2, 3]
for sigma in sigmas:
    filtSize = 2*sigma + 1

    # Gaussian filter
    out = np.zeros_like(noisy)
    for channel in [0, 1, 2]:
        out[..., channel] = np.zeros_like(noisy[..., 0]) # TODO your code here
        out[..., channel] += gaussian(noisy[..., channel], sigma=sigma)
    filtered[f'gaussian_{sigma}'] = out

    # Median filter
    out = np.zeros_like(noisy)
    for channel in [0, 1, 2]:
        out[..., channel] = np.zeros_like(noisy[..., 0]) # TODO your code here
        out[..., channel] += median_filter(noisy[..., channel],size=filtSize)
    filtered[f'median_{filtSize}'] = out

    # Bilateral Filter
    sigmaIntensity = 0.25
    bilateral = np.zeros_like(noisy)
    for channel in [0, 1, 2]:
        bilateral[..., channel] = bilateral2d(noisy[..., channel],
                                              radius=int(sigma),
                                              sigma=sigma,
                                              sigmaIntensity=sigmaIntensity)
    filtered[f'bilateral_{sigma}'] = bilateral


    # Non-local means
    # May take some time - crop the noisy image if you want to debug faster
    nlmSigma = 0.1  # Feel free to modify
    searchWindowRadius = 5
    averageFilterRadius = int(sigma)
    nlm = np.zeros_like(noisy)
    for channel in [0,1,2]:
        nlm[...,channel:channel+1] = nonlocalmeans(noisy[..., channel:channel+1],
                                                   searchWindowRadius,
                                                   averageFilterRadius,
                                                   sigma,
                                                   nlmSigma)
    filtered[f'nlm_{sigma}'] = nlm

# Sample plotting code, feel free to modify!
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(30, 30))
ax[0,0].imshow(clean)
ax[0,0].set_title('Original')
ax[0,0].axis('off')
ax[0,1].imshow(noisy)
ax[0,1].set_title('Noisy')
ax[0,1].axis('off')
for r, sigma in enumerate(sigmas):
    # Plot Gaussian
    ax[r+1,0].imshow(filtered[f'gaussian_{sigma}'])
    ax[r+1,0].set_title(f'Gaussian (sigma={sigma})')
    ax[r+1,0].axis('off')

    # Plot Median
    filtSize = 2 * sigma + 1
    ax[r+1,1].imshow(filtered[f'median_{filtSize}'])
    ax[r+1,1].set_title(f'Median (filter size = {filtSize})')
    ax[r+1,1].axis('off')

    # Plot Bilateral
    ax[r+1,2].imshow(filtered[f'bilateral_{sigma}'])
    ax[r+1,2].set_title(f'Bilateral filter (sigmaSpatial={sigma}, sigmaNlm={nlmSigma})')
    ax[r+1,2].axis('off')

    # Plot Non-local means
    ax[r+1,3].imshow(filtered[f'nlm_{sigma}'])
    ax[r+1,3].set_title(f'Non-local means (sigmaSpatial={sigma}, sigmaNlm={nlmSigma}, searchWindowRadius={searchWindowRadius})')
    ax[r+1,3].axis('off')
ax[0,2].remove()
ax[0,3].remove()
fig.savefig('task3_denoising.pdf', bbox_inches='tight')
