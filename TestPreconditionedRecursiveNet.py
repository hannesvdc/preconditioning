import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import matplotlib
import matplotlib.pyplot as plt

from TrainPreconditionedRecursiveNet import setupRecNet

def testbRecNet():
    # Setup the network and load the weights
    net, _, _ = setupRecNet()
    weights = np.array([ 0.61951713, -1.165711 ,   0.05086731 , 0.57154565, -0.26976452 ,-1.3864234,
                        -1.60178991 , 1.02056758, -1.35905148, -0.93128832]) # Adam + BFGS refinement
    P = net.P
    
    # Generate test data. Same distribution as training data. Test actual training data next
    A_data = net.A_data
    b_data = net.b_data
    N_data = A_data.shape[2]

    # Run each rhs through the neural network
    n_outer_iterations = 10 # Does not need be the same as the number the network was trained on.
    n_inner_iterations = 4
    errors = np.zeros((N_data, n_outer_iterations+1))
    for n in range(N_data):
        A = A_data[:,:,n]
        rhs = b_data[:,n]
        samples = net.forward(weights, A, rhs, n_outer_iterations)
        print(lg.norm(np.matmul(A, P)))

        for k in range(len(samples)):
            err = lg.norm(A.dot(samples[k]) - rhs)
            errors[n,k] = err

    # Solve the system Ax = rhs with gmres for all rhs
    gmres_errors = np.zeros((N_data, n_outer_iterations+1))
    for n in range(N_data):
        rhs = b_data[:,n]
        gmres_errors[n,0] = lg.norm(rhs)

        for k in range(1, n_outer_iterations+1):
            x, _ = slg.gmres(A, rhs, x0=np.zeros(rhs.size), maxiter=k, restart=n_inner_iterations, tol=0.0, M=P)
            gmres_errors[n,k] = lg.norm(A.dot(x) - rhs)

    # Average the errors
    avg_errors = np.average(errors, axis=0)
    avg_gmres_ = np.average(gmres_errors, axis=0)

    # Plot the errors
    fig, ax = plt.subplots()  
    k_axis = np.linspace(0, n_outer_iterations, n_outer_iterations+1)
    rect = matplotlib.patches.Rectangle((net.outer_iterations+0.5, 1.e-16), 7.5, 70, color='gray', alpha=0.2)
    ax.add_patch(rect)
    plt.semilogy(k_axis, avg_errors, label='R2N2 Test Error', linestyle='--', marker='d')
    plt.semilogy(k_axis, avg_gmres_, label='GMRES Error', linestyle='-', marker='^')
    plt.xticks(np.linspace(0, n_outer_iterations, n_outer_iterations+1))
    plt.xlabel(r'# Outer Iterations')
    plt.ylabel('Error')
    plt.xlim((-0.5, 10.5))
    plt.ylim((1.e-16, 70))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    testbRecNet()