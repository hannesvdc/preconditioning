import sys
sys.path.append('../')

import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd

from autograd import jacobian

import internal.Adam as adam

def quadraticTest():
    print('\nRunning Quadratic Test Case.')
    f  = lambda x: 0.5 * (x - 3.0)**2
    df = lambda x: x - 3.0

    optimizer = adam.AdamOptimizer(learning_rate=0.1)

    x0 = 10.0
    xs = optimizer.optimize(df, x0)

    print('Local Minimum', xs, df(xs))

def quadraticTest3d():
    print('\nRunning 3D Quadratic Test Case.')
    rng = rd.RandomState()
    A = rng.normal(size=(3,3))
    b = rng.normal(size=3)
    lmbda = 5.0

    # M needs to be positive definite, so symmetric and positive eigenvalues
    M = np.matmul(np.transpose(A), A) + lmbda*np.eye(3)
    f  = lambda x: 0.5*np.dot(x, np.dot(M, x)) + np.dot(b, x)
    df = lambda x: np.dot(M, x) + b

    optimizer = adam.AdamOptimizer(learning_rate=0.1)

    x0 = np.zeros(3)
    xs = optimizer.optimize(df, x0)

    print('Local Minimum', xs, df(xs))

def multiModal2d():
    print('\nRunning 2D Multi Modal Test Case.')
    f = lambda x: 0.5*((x[0]-1)**2 - 1.5**2)**2 + 0.5*(x[1] + 1)**2 # (-0.5, -1) and (2.5, -1)
    df = jacobian(f)

    optimizer = adam.AdamOptimizer(learning_rate=0.1)

    x0 = np.array([-2.0, 1.0])
    xs = optimizer.optimize(df, x0)

    print('Local Minimum', xs, df(xs))

if __name__ == '__main__':
    quadraticTest()
    quadraticTest3d()
    multiModal2d()