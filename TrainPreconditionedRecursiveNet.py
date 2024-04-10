import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.sparse.linalg as slg

from autograd import jacobian

import api.RecursiveNet as recnet
import api.Scheduler as sch
import algorithms.Adam as adam
import algorithms.BFGS as bfgs


# General setup routine shared by all training routines
def setupRecNet(outer_iterations=3, inner_iterations=4, baseweight=4.0):
    directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/'
    b_filename = 'b_Training_Data.npy'
    b_data = np.load(directory + b_filename)
    N_data = b_data.shape[1]

    A = np.array([[1.392232, 0.152829, 0.088680, 0.185377, 0.156244],
                       [0.152829, 1.070883, 0.020994, 0.068940, 0.141251],
                       [0.088680, 0.020994, 0.910692,-0.222769, 0.060267],
                       [0.185377, 0.068940,-0.222769, 0.833275, 0.058072],
                       [0.156244, 0.141251, 0.060267, 0.058072, 0.735495]])
    A_data = np.repeat(A[:,:,np.newaxis], N_data, axis=2)
    P = slg.spilu(A)

    # Setup classes for training
    net = recnet.R2N2(A_data, b_data, outer_iterations, inner_iterations, P=P, baseweight=baseweight)
    f = lambda w: net.loss(w)
    df = jacobian(f)

    return net, f, df

def sampleWeights(net):
    rng = rd.RandomState()
    inner_iterations = net.inner_iterations
    n_weights = (inner_iterations * (inner_iterations + 1) ) // 2

    while True:
        weights = rng.normal(size=n_weights)
        loss = net.loss(weights)
        if loss < 1.e6:
            return weights
        

def trainRecNetAdam():
    net, f, df = setupRecNet(outer_iterations=3, inner_iterations=4)
    weights = sampleWeights(net)
    print('Initial Loss', f(weights))
    print('Initial Loss Derivative', lg.norm(df(weights)))

    # Setup the optimizer
    scheduler = sch.PiecewiseConstantScheduler({0: 1.e-2, 100: 1.e-3, 3000: 1.e-5, 15000: 1.e-6})
    optimizer = adam.AdamOptimizer(f, df, scheduler=scheduler)
    print('Initial weights', weights)

    # Do the training
    epochs = 20000
    weights = optimizer.optimize(weights, n_epochs=epochs)
    losses = np.array(optimizer.losses)
    grad_norms = np.array(optimizer.gradient_norms)
    print('Done Training. Weights =', weights)

    # Post-processing
    x_axis = np.arange(len(losses))
    plt.semilogy(x_axis, grad_norms, label='Gradient Norms')
    plt.semilogy(x_axis, losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.title('Adam')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    trainRecNetAdam()