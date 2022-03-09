import numpy as np
import skimage.io as io
from skimage import metrics,color
from scipy import ndimage,signal,interpolate
import matplotlib.pyplot as plt


raw = io.imread('lighthouse.png').astype(float)/255
# print(raw.shape)
noisy = io.imread('lighthouse_RAW_noisy_sigma0.01.png').astype(float)/255
L = noisy.shape[0]
W = noisy.shape[1]
red_index = np.zeros((L,W))
green_index = np.ones((L,W))
blue_index = np.zeros((L,W))
red_index[0:L:2,0:W:2] = 1
blue_index[1:L:2,1:W:2] = 1
green_index[0:L:2,0:W:2] = 0
green_index[1:L:2,1:W:2] = 0

# ============== Task 2 Process 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Sampling
red_sample = noisy[red_index==1].reshape((int(L/2),int(W/2)))
blue_sample = noisy[blue_index==1].reshape((int(L/2),int(W/2)))

# Interpolate GREEN
green = noisy*green_index
green += (np.roll(green,1) +
         np.roll(green,-1) +
         np.roll(green,1,axis=0) +
         np.roll(green,-1,axis=1)
         )*0.25

# Interpolate RED and BLUE
y = np.arange(L/2)*2
x = np.arange(W/2)*2
f_red = interpolate.interp2d(x,y,red_sample)
f_blue = interpolate.interp2d(x+1,y+1,blue_sample)
y = np.arange(L)
x = np.arange(W)
red = f_red(x,y)
blue = f_blue(x,y)
print(blue_sample.shape)

fig21 = np.stack([red, green, blue], axis=2)
fig21 = fig21**(1/2.2)


# ============== Task 2 Process 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
ycbcr = color.rgb2ycbcr(fig21)
ycbcr[:,:,1] = ndimage.median_filter(ycbcr[:,:,1], size = 9)
ycbcr[:,:,2] = ndimage.median_filter(ycbcr[:,:,2], size = 9)
fig22 = color.ycbcr2rgb(ycbcr)


# ============== Task 2 Process 3 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

kernel1 = np.array([
    [0, 0, -1, 0, 0],
    [0, 0, 2, 0, 0],
    [-1, 2, 4, 2, -1],
    [0, 0, 2, 0, 0],
    [0, 0, -1, 0, 0]
])
kernel2 = np.array([
    [0, 0, 1/2, 0, 0],
    [0, -1, 0, -1, 0],
    [-1, 4, 5, 4, -1],
    [0, -1, 0, -1, 0],
    [0, 0, 1/2, 0, 0]
])
kernel3 = np.array([
    [0, 0, -1, 0, 0],
    [0, -1, 4, -1, 0],
    [1/2, 0, 5, 0, 1/2],
    [0, -1, 4, -1, 0],
    [0, 0, -1, 0, 0]
])
kernel4 = np.array([
    [0, 0, -3/2, 0, 0],
    [0, 2, 0, 2, 0],
    [-3/2, 0, 6, 0, -3/2],
    [0, 2, 0, 2, 0],
    [0, 0, -3/2, 0, 0]
])

conv1 = signal.convolve2d(noisy,kernel1,mode='same')
conv2 = signal.convolve2d(noisy,kernel2,mode='same')
conv3 = signal.convolve2d(noisy,kernel3,mode='same')
conv4 = signal.convolve2d(noisy,kernel4,mode='same')

rinrb_inedx = np.zeros_like(green_index)+green_index
rinrb_inedx[1:L:2,:] = 0
rinbr_inedx = np.zeros_like(green_index)+green_index
rinbr_inedx[0:L:2,:] = 0

green = noisy * green_index + conv1*red_index/8 + conv1*blue_index/8
red = noisy * red_index + conv2*rinrb_inedx/8 + conv3*rinbr_inedx/8 + conv4*blue_index/8
blue = noisy * blue_index + conv2*rinbr_inedx/8 + conv3*rinrb_inedx/8 + conv4*red_index/8

fig23 = np.stack([red, green, blue], axis=2)
fig23 = (fig23*(fig23>0))**(1/2.2)

# ============== Result >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
io.imsave('task2_1.png',fig21)
io.imsave('task2_2.png',fig22)
io.imsave('task2_3.png',fig23)

print('Process 1 PSNR:',metrics.peak_signal_noise_ratio(raw,fig21))
print('Process 2 PSNR:',metrics.peak_signal_noise_ratio(raw,fig22))
print('Process 3 PSNR:',metrics.peak_signal_noise_ratio(raw,fig23))

plt.figure(figsize=(100,50))
plt.subplot(1,3,1)
plt.imshow(fig21)
plt.axis(False)
plt.subplot(1,3,2)
plt.imshow(fig22)
plt.axis(False)
plt.subplot(1,3,3)
plt.imshow(fig23)
plt.axis(False)
plt.savefig('contrast.pdf')
