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

    optimizer = adam.AdamOptimizer(learning_rate=0.01)

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

def MullerBrown():
    print('\nRunning Muller-Brown Test Case')
    A=[-200,-100,-170,15]
    a=[-1,-1,-6.5,0.7]
    b=[0,0,11,0.6]
    c=[-10,-10,-6.5,0.7]
    x0=[1,0,-0.5,-1]
    y0=[0,0.5,1.5,1]

    def V(u):
        p = 0.0
        for k in range(4):
            t1 = a[k]*(u[0] - x0[k])**2
            t2 = b[k]*(u[0] - x0[k])*(u[1] - y0[k])
            t3 = c[k]*(u[1] - y0[k])**2
            p += A[k]*np.exp(t1 + t2 + t3)
        return p
    dV = jacobian(V)

    rng = rd.RandomState()
    optimizer = adam.AdamOptimizer(learning_rate=0.01)

    u0 = rng.normal(np.array([0.0, 0.6]), scale=np.sqrt(0.5), size=2)
    us = optimizer.optimize(dV, u0)

    print('Local Minimum', us, dV(us))

def Ackleys():
    print('\nRunning Ackley\'s Test Case')
    f = lambda x: -20.0*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(2.0*np.pi*x[0]) + np.cos(2.0*np.pi*x[1])))
    df = jacobian(f)

    rng = rd.RandomState()
    optimizer = adam.AdamOptimizer(learning_rate=0.1)

    u0 = rng.normal(scale=2.0, size=2)
    us = optimizer.optimize(df, u0)

    print('Local Minimum', us, df(us))

if __name__ == '__main__':
    quadraticTest()
    quadraticTest3d()
    multiModal2d()
    MullerBrown()
    Ackleys()