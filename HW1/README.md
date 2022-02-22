# Working with Python in EE 367

## Setup
### miniconda
We recommend (and will only support) using the
[anaconda](https://www.anaconda.com) framework for managing your python
installation. You can install miniconda (a lightweight version of anaconda) by
going to [this page](https://docs.conda.io/en/latest/miniconda.html),
downloading, and installing the appropriate package for your system.

### conda environment
You can set up a Python 3.7 environment named `ee367-hw` by running

``` sh
conda env create -f environment.yml 
```
This will install Python 3.7 and a small number of packages needed to get up and
running. You can then activate the environment by running the command

``` sh
conda activate ee367-hw
```
Once your environment is active, any scripts you write will have access to the
required packages for this class. When you're done with your work, you can run

``` sh
conda deactivate
```
to return to the base conda environment.

If you need to reinstall this environment, first remove it as
``` sh
conda remove --name ee367-hw --all
```
and then reinstall it as described above

## Plots and IO

We recommend using the `matplotlib` package for creating figures. For saving
images directly to disk, we recommend using the command

``` python
import skimage.io as io
io.imsave(<path to image>, img)
```
For saving the current figure, you can use

``` python
import matplotlib.pyplot as plt

plt.savefig(<path to file>)
```

For those familiar with the matlab syntax for creating subplots, an analogous
thing can be done in python:

``` python
fig, axs = plt.subplots(nrows=<num rows>, ncols=<num cols>)
axs[0,1].imshow(img)
axs[0,1].set_title("My image")
axs[1,0].plot(data)
axs[1,0].set_xlabel("Time (s)")
...
```

If you're having trouble getting your images to look right, you can try one or
more of the following:
- Make sure your image is of shape `[height, width, num_channels]` and that the
  channels are in RGB order.
- Use a combination of `np.clip`, `.astype(np.uint8)`, `.astype(np.float64)`, and other shenanigans to
  get your image into the correct range (generally [0, 255] for int-based images,
  [0, 1] for floating point images)
- Use `plt.imshow(img)`, which will automatically perform some computation to make
  the image look reasonable.

Written by: Mark Nishimura 
Last edited by: Gordon Wetzstein 09/02/2021

