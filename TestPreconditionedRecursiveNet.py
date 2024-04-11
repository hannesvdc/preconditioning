import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import matplotlib
import matplotlib.pyplot as plt

import TrainRecursiveNet
import TrainPreconditionedRecursiveNet

## This function combines convergence rates of preconditioned and non-preconditioned R2N2 / GMRES
def testbRecNet():
    # Setup the network and load the weights
    net, _, _ = TrainRecursiveNet.setupRecNet()
    precond_net, _, _ = TrainPreconditionedRecursiveNet.setupRecNet()
    weights = np.array([-0.75338897, -0.69512928 ,-1.28298784, -0.8088681 ,  0.87909088 , 1.48089791,
                        -0.69293485 ,-2.75058767 ,-2.00362172 , 0.90054142]) # Adam + BFGS refinement fixed A
    precond_weights = np.array([-1.18614796 ,-0.16825594, -0.05674233 , 1.21868203 ,-0.557387   ,-0.78371623,
                        -0.11607332 , 2.58240297 , 0.58406379, -0.53851497]) # Adam + BFGS refinement
    P = precond_net.P
    
    # Generate test data. Same distribution as training data. Test actual training data next
    A_data = precond_net.A_data
    b_data = precond_net.b_data
    N_data = A_data.shape[2]

    # Run each rhs through the neural network
    n_outer_iterations = 10 # Does not need be the same as the number the network was trained on.
    n_inner_iterations = 4
    precond_errors = np.zeros((N_data, n_outer_iterations+1))
    errors = np.zeros((N_data, n_outer_iterations+1))
    for n in range(N_data):
        A = A_data[:,:,n]
        rhs = b_data[:,n]
        precond_samples = precond_net.forward(precond_weights, A, rhs, n_outer_iterations)
        samples = net.forward(weights, A, rhs, n_outer_iterations)

        for k in range(len(samples)):
            precond_err = lg.norm(A.dot(precond_samples[k]) - rhs)
            err = lg.norm(A.dot(samples[k]) - rhs)
            precond_errors[n,k] = precond_err
            errors[n,k] = err

    # Solve the system Ax = rhs with gmres for all rhs
    precond_gmres_errors = np.zeros((N_data, n_outer_iterations+1))
    gmres_errors = np.zeros((N_data, n_outer_iterations+1))
    for n in range(N_data):
        rhs = b_data[:,n]
        precond_gmres_errors[n,0] = lg.norm(rhs)
        gmres_errors[n,0] = lg.norm(rhs)

        for k in range(1, n_outer_iterations+1):
            x, _ = slg.gmres(A, rhs, x0=np.zeros(rhs.size), maxiter=k, restart=n_inner_iterations, tol=0.0, M=P)
            precond_gmres_errors[n,k] = lg.norm(A.dot(x) - rhs)
            x, _ = slg.gmres(A, rhs, x0=np.zeros(rhs.size), maxiter=k, restart=n_inner_iterations, tol=0.0)
            gmres_errors[n,k] = lg.norm(A.dot(x) - rhs)

    # Average the errors
    avg_precond_errors = np.average(precond_errors, axis=0)
    avg_errors = np.average(errors, axis=0)
    avg_precond_gmres_ = np.average(precond_gmres_errors, axis=0)
    avg_gmres_ = np.average(gmres_errors, axis=0)

    # Plot the errors
    fig, ax = plt.subplots()  
    k_axis = np.linspace(0, n_outer_iterations, n_outer_iterations+1)
    rect = matplotlib.patches.Rectangle((net.outer_iterations+0.5, 1.e-16), 7.5, 70, color='gray', alpha=0.2)
    ax.add_patch(rect)
    plt.semilogy(k_axis, avg_precond_errors, label='Preconditioned R2N2 Test Error', linestyle='--', marker='d')
    plt.semilogy(k_axis, avg_errors, label='R2N2 Test Error', linestyle='--', marker='d')
    plt.semilogy(k_axis, avg_precond_gmres_, label='Preconditioned GMRES Error', linestyle='-', marker='^')
    plt.semilogy(k_axis, avg_gmres_, label='GMRES Error', linestyle='-', marker='^')
    plt.xticks(np.linspace(0, n_outer_iterations, n_outer_iterations+1))
    plt.xlabel(r'# Outer Iterations')
    plt.ylabel('Error')
    plt.xlim((-0.5, 10.5))
    plt.ylim((1.e-16, 70))
    plt.title(r'Random $b$, Fixed $A$, $P$ = ILU(0)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    testbRecNet()