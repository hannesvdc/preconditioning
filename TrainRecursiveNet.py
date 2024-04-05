import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import matplotlib.pyplot as plt
import scipy.optimize as opt

from autograd import jacobian

import api.RecursiveNet as recnet
import api.Scheduler as sch
import algorithms.Adam as adam
import algorithms.BFGS as bfgs

# General setup routine shared by all training routines
def setupRecNet(outer_iterations=3, inner_iterations=4):

    # Sample Data
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
    net = recnet.R2N2(A, outer_iterations, inner_iterations, b)
    f = lambda w: net.loss(w)
    df = jacobian(f)

    return net, f, df

def sampleWeights(net):
    rng = rd.RandomState()
    inner_iterations = net.inner_iterations
    n_weights = (inner_iterations * (inner_iterations + 1) ) // 2

    while True:
        weights = rng.normal(size=n_weights)
        if net.loss(weights) < 5000:
            return weights

def trainRecNetAdam():
    net, f, df = setupRecNet()
    weights = sampleWeights(net)
    print('Initial Loss', f(weights))
    print('Initial Loss Derivative', lg.norm(df(weights)))

    # Setup the optimizer
    scheduler = sch.PiecewiseConstantScheduler({0: 0.01, 1000: 0.001, 5000: 1.e-4, 10000: 1.e-5})
    optimizer = adam.AdamOptimizer(f, df, scheduler=scheduler)
    print('Initial weights', weights)

    # Do the training
    epochs = 15000
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

def trainRecNetBFGS():
    net, f, df = setupRecNet()
    weights = sampleWeights(net)
    print('Initial Loss', f(weights))
    print('Initial Loss Derivative', lg.norm(df(weights)))

    losses = []
    grad_norms = []
    epoch_counter = [0]
    def callback(x):
        print('\nEpoch #', epoch_counter[0])
        l = f(x)
        g = lg.norm(df(x))
        losses.append(l)
        grad_norms.append(g)
        epoch_counter[0] += 1
        print('Loss =', l)
        print('Gradient Norm =', g)
        print('Weights', x)

    epochs = 5000
    method = 'L-BFGS-B'
    result = opt.minimize(f, weights, jac=df, method=method,
                                              options={'maxiter': epochs, 'gtol': 1.e-100}, 
                                              callback=callback)
    weights = result.x
    print('Minimzed Loss', f(weights), df(weights))
    print('Minimization Result', result)

    # Post-processing
    x_axis = np.arange(len(losses))
    plt.semilogy(x_axis, grad_norms, label='Gradient Norms')
    plt.semilogy(x_axis, losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.title(method)
    plt.legend()
    plt.show()

def refineRecNet(): # Train NN with own bfgs implementation
    _, f, df = setupRecNet()
    weights = np.load()
    
    print('Initial Loss', f(weights))
    print('Initial Loss Derivative', lg.norm(df(weights)))

    learning_rate = 0.01
    optimizer = bfgs.BFGSOptimizer(f, df, scheduler=sch.ConstantScheduler(learning_rate))

    epochs = 5000
    weights = optimizer.optimize(weights, maxiter=epochs)
    losses = optimizer.losses
    grad_norms = optimizer.gradient_norms
    print('Weights', weights)
    print('Minimzed Loss', f(weights), df(weights))

    # Post-processing
    x_axis = np.arange(len(losses))
    plt.semilogy(x_axis, grad_norms, label='Gradient Norms')
    plt.semilogy(x_axis, losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.title('BFGS Refinement with learning rate = ' + str(learning_rate))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    trainRecNetAdam()