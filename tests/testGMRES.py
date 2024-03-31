import sys
sys.path.append('../')

import autograd.numpy as np
import autograd.numpy.linalg as lg

from algorithms.GMRES import *

def testConvergence():
    n = 4
    A = np.random.rand(n, n)
    b = np.random.rand(n)
    x0 = np.ones(n)

    action = lambda x: np.dot(A, x)
    max_it = 100
    tolerance = 1.e-12
    solution, residue = GMRES(action, b, x0, max_it, tolerance)

    print('Solution:', solution)
    print('Residue:', residue, lg.norm(action(solution) - b))

if __name__ == '__main__':
    testConvergence()