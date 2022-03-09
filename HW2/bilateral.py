import numpy as np
from fspecial import fspecial_gaussian_2d

def bilateral2d(img, radius, sigma, sigmaIntensity):
    pad = radius
    # Initialize filtered image to 0
    out = np.zeros_like(img)

    # Pad image to reduce boundary artifacts
    imgPad = np.pad(img, pad)

    # Smoothing kernel, gaussian with standard deviation sigma
    # and size (2*radius+1, 2*radius+1)
    filtSize = (2*radius + 1, 2*radius + 1)
    spatialKernel = fspecial_gaussian_2d(filtSize, sigma)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            centerVal = imgPad[y+pad, x+pad] # Careful of padding amount!

            # Go over a window of size (2*radius + 1) around the current pixel,
            # compute weights, sum the weighted intensity.
            # Don't forget to normalize by the sum of the weights used.

            # TODO: Replace with your own code
            vicinity = imgPad[y:y+2*pad+1,x:x+2*pad+1]
            intensity_weight = np.exp(-(vicinity - centerVal)**2/(2*sigmaIntensity))
            weight = intensity_weight * spatialKernel
            W_p = np.sum(weight)
            out[y, x] = np.sum(vicinity * weight) / W_p

    return out
