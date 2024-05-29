import autograd.numpy as np
import autograd.numpy.linalg as lg

import scipy.sparse.linalg as slg

from algorithms.GMRES import *

def NewtonKrylov(psi, d_psi, x0, max_it, tolerance=1.e-6, verbose=False):
    s = x0.size
    x = np.copy(x0)
    f = psi(x)
    print('\nInitial Residue', lg.norm(f))

    num_it = 0
    while num_it <= max_it and lg.norm(f) > tolerance:
        if verbose:
            print('\nNK Iteration', num_it, ', Currrent Residue:', lg.norm(f))

        # Setup operators for GMRES
        dpsi_v = lambda v: d_psi(x, v, f)
        A = slg.LinearOperator((s, s), matvec=dpsi_v)
        def cb(input):
            if verbose:
                print('GMRES Residue', lg.norm(A.matvec(input) + f))

        # Use the built-in GMRES routine, my implementation didn't cut it
        dx, _ = slg.lgmres(A, -f, np.copy(x), callback=cb, maxiter=1, outer_v=[], store_outer_Av=False)
        gmres_res = lg.norm(A.matvec(dx) + f)

        # Update the Newton Iterate
        x = x + dx
        f = psi(x)
        if verbose:
            print('Residue After GMRES', gmres_res, lg.norm(f))

        num_it += 1

    if num_it >= max_it:
        print('Newton-Krylov did not Convergence withing the Specified Number of Iterations. Returning Curren Iteration.')
        return x, f
    
    print('\nNK Solution Found After',  num_it, 'Iterations!\n')
    return x, f