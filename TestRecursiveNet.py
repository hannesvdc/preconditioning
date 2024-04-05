import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import matplotlib.pyplot as plt

from TrainRecursiveNet import setupRecNet

def testRecNet():
    # Setup the network and load the weights
    net, _, _ = setupRecNet()
    #weights = np.array([-5.29079523e-02, -4.26651859e-02, -7.89517538e+00,  9.57357343e-01,
    #                    -6.60445132e-01,  9.73932767e+01, -5.17671502e-02, -2.90171257e+00,
    #                    -4.06082443e-01,  1.35692637e-03]) # BFGS weights
    weights = np.array([-0.01424995, -0.01421451, -0.88593378, -0.01427751,  0.86512796  ,
                        1.39955378,  -0.01422662, -2.53569552, -1.6169461,   0.84363141]) # Adam Weights
    
    # Generate test data. Same distribution as training data. Test actual training data next
    N_data = 1000
    rng = rd.RandomState()
    A = np.array([[1.392232, 0.152829, 0.088680, 0.185377, 0.156244],
                  [0.152829, 1.070883, 0.020994, 0.068940, 0.141251],
                  [0.088680, 0.020994, 0.910692,-0.222769, 0.060267],
                  [0.185377, 0.068940,-0.222769, 0.833275, 0.058072],
                  [0.156244, 0.141251, 0.060267, 0.058072, 0.735495]])
    b_mean = np.array([2.483570, -0.691321, 3.238442, 7.615149, -1.170766])
    b_mean_repeated = np.array([b_mean,]*N_data).transpose()
    b = b_mean_repeated + rng.uniform(low=-1, high=1, size=(5, N_data))

    # Run each rhs through the neural network
    n_outer_iterations = 10 # Does not need be the same as the number the network was trained on.
    n_inner_iterations = 3
    errors = np.zeros((N_data, n_outer_iterations+1))
    for n in range(N_data):
        rhs = b[:,n]
        samples = net.forward(weights, rhs, n_outer_iterations)

        for k in range(len(samples)):
            err = lg.norm(A.dot(samples[k]) - rhs)
            errors[n,k] = err

    # Solve the system Ax = rhs with gmres for all rhs
    gmres_errors = np.zeros((N_data, n_outer_iterations+1))
    for n in range(N_data):
        rhs = b[:,n]
        gmres_errors[n,0] = lg.norm(rhs)

        for k in range(1, n_outer_iterations+1):
            x, _ = slg.gmres(A, rhs, x0=np.zeros(rhs.size), maxiter=k, restart=n_inner_iterations, tol=0.0)
            gmres_errors[n,k] = lg.norm(A.dot(x) - rhs)

    # Average the errors
    avg_errors = np.average(errors, axis=0)
    avg_gmres_ = np.average(gmres_errors, axis=0)

    # Plot the errors
    k_axis = np.linspace(0, n_outer_iterations, n_outer_iterations+1)
    plt.semilogy(k_axis, avg_errors, label='R2N2 Test Error', linestyle='--', marker='d')
    plt.semilogy(k_axis, avg_gmres_, label='GMRES Error', linestyle='-', marker='^')
    plt.xticks(np.linspace(0, n_outer_iterations, n_outer_iterations+1))
    plt.xlabel(r'# Outer Iterations')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    testRecNet()

