import numpy as np
from tqdm import tqdm
from time import time


def grad_l2(A, x, b):
    # TODO: return the gradient of 0.5 * ||Ax - b||_2^2
    return None


def residual_l2(A, x, b):
    return 0.5 * np.linalg.norm(A @ x - b)**2


def run_gd(A, b, step_size=1e-4, num_iters=1500, grad_fn=grad_l2, residual=residual_l2):
    ''' Run gradient descent to solve Ax = b

    Parameters
    ----------
    A : matrix of size (N_measurements, N_dim)
    b : observations of (N_measurements, 1)
    step_size : gradient descent step size
    num_iters : number of iterations of gradient descent
    grad_fn : function to compute the gradient
    residual : function to compute the residual

    Returns
    -------
    x
        output matrix of size (N_dim)

    residual
        list of calculated residuals at each iteration

    timing
        time to execute each iteration (should be cumulative to each iteration)

    '''

    # initialize x to zero
    x = np.zeros((A.shape[1], 1))

    # TODO: complete the gradient descent algorithm here
    # you can also complete and use the grad_l2 and residual_l2 functions

    # this function can return a list of residuals and timings at each iteration so
    # you can plot them and include them in your report

    # don't forget to also implement the stochastic gradient descent version of this function!


def run_lsq(A, b):
    ''' Numpy's implementation of least squares which uses SVD & matrix factorization from LAPACK '''
    x, resid, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x, 0.5 * resid


if __name__ == '__main__':

    # set up problem
    N_dim = 128  # size of x
    N_measurements = 16384  # number of measurements or rows of A

    # load matrix
    dat = np.load('task3.npy', allow_pickle=True)[()]

    # data matrix -- here the rows are measurements and columns are dimension N_dim
    A = dat['A']

    # corrupted measurements
    b = dat['b']

    # least squares solve using SVD + matrix factorization
    # you can compare your solution to this
    x_lsq, resid_lsq = run_lsq(A, b)

    # TODO: implement and call your GD and SGD functions
    # run_gd()
