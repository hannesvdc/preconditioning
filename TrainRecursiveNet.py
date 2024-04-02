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
    learning_rate = 0.01
    net = recnet.R2N2(A, outer_iterations, inner_iterations, b)
    f = lambda w: net.loss(w)
    df = jacobian(f)
    optimizer = adam.AdamOptimizer(f, df, learning_rate=learning_rate) # Adam optimizer with standard parameters

    # Setup the weights as a vector with 10 elements (not a lower-triangular matrix because we need to take gradients)
    n_weights = (inner_iterations * (inner_iterations + 1) ) // 2
    weights = rng.normal(size=n_weights)
    print('Initial weights', weights)

    # Do the training
    epochs = 10000
    print('Initial Loss', f(weights))
    print('Initial Loss Derivative', lg.norm(df(weights)))
    weights = optimizer.optimize(weights, n_epochs=1000)
    optimizer.setLearningRate(0.001)
    weights = optimizer.optimize(weights, n_epochs=5000) # Continue for the remaining epochs-500 iterations
    optimizer.setLearningRate(0.0001)
    weights = optimizer.optimize(weights, n_epochs=epochs-6000)
    losses = np.array(optimizer.losses)
    grad_norms = np.array(optimizer.gradient_norms)
    print('Done Training')

    # Post-processing
    x_axis = np.arange(len(losses))
    plt.semilogy(x_axis, losses, label='Training Loss')
    plt.semilogy(x_axis, grad_norms, label='Gradient Norms')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    trainRecNet()