import numpy as np
from fspecial import fspecial_gaussian_2d

def inbounds(img, y, x):
    return 0 <= y and y < img.shape[0] and \
           0 <= x and x < img.shape[1]

def comparePatches(patch1, patch2, kernel, sigma):
    return np.exp(-np.sum(kernel*(patch1 - patch2) ** 2)/(2*sigma**2))

def nonlocalmeans(img, searchWindowRadius, averageFilterRadius, sigma, nlmSigma):
    # Initialize output to 0
    out = np.zeros_like(img)
    # Pad image to reduce boundary artifacts
    pad = max(averageFilterRadius, searchWindowRadius)
    imgPad = np.pad(img, pad)
    imgPad = imgPad[..., pad:-pad] # Don't pad third channel

    # Smoothing kernel
    filtSize = (2*averageFilterRadius + 1, 2*averageFilterRadius + 1)
    kernel = fspecial_gaussian_2d(filtSize, sigma)
    # Add third axis for broadcasting
    kernel = kernel[:, :, np.newaxis]
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            centerPatch = imgPad[y+pad-averageFilterRadius:y+pad+averageFilterRadius+1,
                                 x+pad-averageFilterRadius:x+pad+averageFilterRadius+1,
                                 :]
            # Go over a window around the current pixel, compute weights
            # based on difference of patches, sum the weighted intensity
            # Hint: Do NOT include the patches centered at the current pixel
            # in this loop, it will throw off the weights
            weights = np.zeros((2*searchWindowRadius+1, 2*searchWindowRadius+1, 1))

            # This makes it a bit better: Add current pixel as well with max weight
            # computed from all other neighborhoods.
            max_weight = 0

            out[y, x, :] = 0. # TODO: Replace with your code.
            sWR = searchWindowRadius
            J = imgPad[
                y + pad - sWR:y + pad + sWR + 1,
                x + pad - sWR:x + pad + sWR + 1,
                :
            ]
            for yy in range(2*sWR+1):
                for xx in range(2*sWR+1):
                    if xx == sWR & yy == sWR:
                        continue
                    x_j = x + xx - sWR + pad
                    y_j = y + yy - sWR + pad
                    if inbounds(img, y_j-averageFilterRadius, x_j-averageFilterRadius):
                        patch = imgPad[y_j-averageFilterRadius:y_j+averageFilterRadius+1,
                                       x_j-averageFilterRadius:x_j+averageFilterRadius+1,
                                       :]
                        weights[yy, xx, 0] = np.sum(comparePatches(patch,centerPatch,kernel,nlmSigma))
                        if weights[yy, xx, 0] > max_weight:
                            max_weight = weights[yy, xx, 0]
            weights[sWR, sWR] = max_weight
            out[y, x, :] = np.sum(weights * J / np.sum(weights))
    return out
