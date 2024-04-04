import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import matplotlib.pyplot as plt

from TrainRecursiveNet import setupRecNet

def testRecNet():

    # Setup the network and load the weights
    net, _, _ = setupRecNet()
    weights = np.load() # Refer to file or add weights in place

    # Generate test data
    N_data = 100
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
    n_outer_iterations = 5 # Does not need be the same as the number the network was trained on.
    errors = np.zeros((N_data, n_outer_iterations))
    for n in range(N_data):
        rhs = b[:,n]
        samples = net.forward(weights, rhs, n_outer_iterations)

        for k in range(len(samples)):
            err = lg.norm(A.dot(samples[k]) - rhs)
            errors[n,k] = err

    # Average and plot the errors
    avg_errors = np.average(errors, axis=0)
    k_axis = np.linspace(1, n_outer_iterations, n_outer_iterations)
    plt.semilogy(k_axis, avg_errors, label='R2N2 Error')
    plt.xlabel(r'$k$')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    testRecNet()

