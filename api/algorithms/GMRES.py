import autograd.numpy as np
import autograd.numpy.linalg as lg

def GMRES(action, rhs, x0, max_it, tolerance=1.e-6, verbose=False):

    # Setting up matrices for Arnoldi Iteration
    Q = np.zeros((rhs.size, max_it))
    H = np.zeros((max_it + 1, max_it))
    r0 = rhs - action(x0)  # b - Ax0
    beta = lg.norm(r0)
    Q[:,0] = r0 / beta

    for j in range(max_it):
        if verbose:
            print('GMRES Iteration', j)
        # Create new Krylov vector and orthonormalize using Gram-Schmidt
        Q[:, j+1] = action(Q[:,j])  # A q_j
        for i in range(j):
            H[i, j]  = np.dot(Q[:,i], Q[:, j+1])
            Q[:,j+1] = Q[:, j+1] - H[i, j]*Q[:,i]
        H[j+1, j] = lg.norm(Q[:, j+1])
        Q[:, j+1] = Q[:,j+1] / H[j+1, j]

        # Solve the least-squares system
        e_1 = np.zeros(j+2)
        e_1[0] = 1.0
        y, res, _, _ = lg.lstsq(H[:(j+2), :(j+1)], beta*e_1, rcond=None)
        print('GMRES Residue', res[0])

        # If we have reached the tolerance, return solution and residual
        if lg.norm(res[0]) < tolerance:
            print('Solution Found in', j, 'Iterations!')
            return np.dot(Q[:,:(j+1)], y) + x0, res[0]

    print('Maxiumum Number of Iterations Reached, Returning.')
    return np.dot(Q[:,:(j+1)], y) + x0, res[0]
