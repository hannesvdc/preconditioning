import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import matplotlib.pyplot as plt

from autograd import jacobian

import api.RecursiveNet as recnet
import algorithms.Adam as adam

def trainRecNet():
    # Generate the training data
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

    # Setup classes for training
    outer_iterations = 3
    inner_iterations = 4
    net = recnet.R2N2(A, outer_iterations, inner_iterations, b)
    optimizer = adam.AdamOptimizer() # Adam optimizer with standard parameters

    # Setup the weights as a vector with 10 elements (not a lower-triangular matrix because we need to take gradients)
    n_weights = (inner_iterations * (inner_iterations + 1) ) // 2
    weights = rng.normal(size=n_weights)
    print('Initial weights', weights)

    # Do the training
    n_epochs = 1
    f = lambda w: net.loss(w)
    df = jacobian(f)
    print(f(weights))
    print(df(weights))
    #optimized_weights = optimizer.optimize(df, weights, n_epochs=n_epochs)

    # Post-processing
    #print('Optimized Weights', optimized_weights)
    #print('Loss', f(optimized_weights))

if __name__ == '__main__':
    trainRecNet()