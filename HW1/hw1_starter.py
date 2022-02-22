import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.fft import fft2, ifft2, fftshift, ifftshift

hw_dir = Path(__file__).parent

# Load images
img1 = io.imread(hw_dir/'image1.png')
img2 = io.imread(hw_dir/'image2.png')
img1 = img1.astype(np.float64)/255
img2 = img2.astype(np.float64)/255

# Part (a)
W = img1.shape[0]       # = 1001 dots
d = np.array([0.4, 2])  # distances (m)
dpi = 300               # dots per inch

#### YOUR CODE HERE ####

# Part (b)
cpd = 5   # Peak contrast sensitivity location (cycles per degree)

#### YOUR CODE HERE ####

# Part (c)
# Hint: fft2, ifft2, fftshift, and ifftshift functions all take an |axes|
# argument to specify the axes for the 2D DFT. e.g. fft2(arr, axes=(1, 2))
# Hint: Check out np.meshgrid.

#### Change these to the correct values for the high- and low-pass filters

size = 30
squre = 0
# Square
if squre:
    lpf = np.zeros_like(img1[:,:,0])
    lpf[(501-size):(501+size), (501-size):(501+size)] = 1
# Circle
else:
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, W)
    xx, yy = np.meshgrid(x, y)
    zz = xx ** 2 + yy ** 2
    lpf = zz < (size/500)**2
hpf = 1 - lpf
#### Apply the filters to create the hybrid image
hybrid_img = np.ones_like(img1)
for i in range(3):
    h1 = fftshift(fft2(img2[:, :, i])) * hpf
    h2 = fftshift(fft2(img1[:, :, i])) * lpf
    hybrid_img[:, :, i] = ifft2( ifftshift( h1 + h2 ))


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
axs[0,0].imshow(img2)
axs[0,0].axis('off')
axs[0,1].imshow(hpf, cmap='gray')
axs[0,1].set_title("High-pass filter")
axs[1,0].imshow(img1)
axs[1,0].axis('off')
axs[1,1].imshow(lpf, cmap='gray')
axs[1,1].set_title("Low-pass filter")
plt.savefig("hpf_lpf.png", bbox_inches='tight')
io.imsave("hybrid_image.png", np.clip(hybrid_img, a_min=0, a_max=255.))
