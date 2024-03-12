import sys
sys.path.append('../')

import autograd.numpy as anp
import autograd.numpy.linalg as alg
import numpy.random as rd
from autograd import jacobian


from NewtonKrylov import *

def testNewtonKrylov1():
    psi = lambda x: np.array([x[0]*np.sin(x[1]), x[0]**2 + x[1]**2 - 1.0])
    _dpsi = lambda x: np.array([[np.sin(x[1]), x[0]*np.cos(x[1])], [2*x[0], 2*x[1]]])
    dpsi = lambda x, v: np.dot(_dpsi(x), v)

    x0 = rd.rand(2)
    max_it = 100
    tolerance = 1.e-10
    solution, f_value = NewtonKrylov(psi, dpsi, x0, max_it, max_it, tolerance=tolerance)

    print('Solution:', solution)
    print('Function Value:', lg.norm(f_value))

def testNewtonKrylov2():
    n = 5
    a = rd.rand(n)
    b = rd.rand(n)

    psi = lambda x: x*np.exp(np.dot(x, a)) - b
    _dpsi = jacobian(psi)
    dpsi = lambda x, v: np.dot(_dpsi(x), v)

    x0 = rd.rand(n)
    max_it = 100
    tolerance = 1.e-12
    solution, f_value = NewtonKrylov(psi, dpsi, x0, max_it, max_it, tolerance=tolerance)

    print('Solution:', solution)
    print('Function Value:', f_value, lg.norm(f_value))

if __name__ == '__main__':
    testNewtonKrylov2()