import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import matplotlib.pyplot as plt
import scipy.optimize as opt

from autograd import jacobian

import api.NewtonKrylovRecursiveNet as recnet
import api.Scheduler as sch
import algorithms.Adam as adam
import algorithms.BFGS as bfgs

# General setup routine shared by all training routines
def setupRecNet(outer_iterations=3, inner_iterations=4, baseweight=4.0):
    # Define the Chandresakhar H-function
    c = 0.875
    m = 10
    I = np.ones(m)
    mu = (np.arange(1, m+1, 1) - 0.5) / m
    def computeAc():
        Ac = np.zeros((10,10))
        for i in range(10):
            for j in range(10):
                Ac[i,j] = mu[i] / (mu[i] + mu[j])
        return 0.5 * c/m * Ac
    Ac = computeAc()
    H = lambda x: x + np.divide(I, I + np.dot(Ac, x))

    # Sample data - the inittial conditions x_0,i, i = data index
    N_data = 1000
    rng = rd.RandomState()
    x0_data = rng.normal(1.0, np.sqrt(0.2), size=(m, N_data))

    # Setup classes for training
    net = recnet.NewtonKrylovSuperStructure(H, x0_data, outer_iterations, inner_iterations, baseweight=baseweight)
    f = lambda w: net.loss(w)
    df = jacobian(f)

    return net, f, df

def sampleWeights(net, threshold=1.e6):
    rng = rd.RandomState()
    inner_iterations = net.inner_iterations
    n_weights = (inner_iterations * (inner_iterations + 1) ) // 2

    print('Selecting Proper Random Initial Condition')
    while True:
        weights = rng.normal(size=n_weights)
        if net.loss(weights) < threshold:
            return weights

def trainNKNetAdam():
    net, f, df = setupRecNet(outer_iterations=3, inner_iterations=10)
    weights = sampleWeights(net, threshold=1.e4)
    print('Initial Loss', f(weights))
    print('Initial Loss Derivative', lg.norm(df(weights)))

    # Setup the optimizer
    scheduler = sch.PiecewiseConstantScheduler({0: 1.e-2, 10: 1.e-3, 1000: 1.e-4, 10000: 1.e-5})
    optimizer = adam.AdamOptimizer(f, df, scheduler=scheduler)
    print('Initial weights', weights)

    # Do the training
    epochs = 5000
    try:
        weights = optimizer.optimize(weights, n_epochs=epochs)
    except KeyboardInterrupt: # If Training has converged well enough with Adam, the user can stop manually
        print('Aborting Training. Plotting Convergence')
    print('Done Training at', len(optimizer.losses), 'epochs. Weights =', weights)
    losses = np.array(optimizer.losses)
    grad_norms = np.array(optimizer.gradient_norms)

    # Post-processing
    x_axis = np.arange(len(losses))
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.semilogy(x_axis, losses, label='Training Loss')
    plt.semilogy(x_axis, grad_norms, label='Loss Gradient')
    plt.xlabel('Epoch')
    plt.title('Adam')
    plt.legend()
    plt.show()

def trainNKNetBFGS():
    _, f, df = setupRecNet(outer_iterations=3, inner_iterations=10)
    #weights = np.array([-1.93181061,  0.18862427 ,-0.36436414,  1.75800653,  0.81753954 ,-2.90153424,
    #                    1.11418358 ,-1.1968051 ,  0.35490947 , 0.77058088]) # Initial weights found by Adamm optimizer + 4 inner iterations
    weights = np.array([-1.24055358 ,-0.4106254  , 0.53113248 ,-2.16433248 , 0.65725721 ,-0.21630475,
                        -1.17453146 , 2.24363514,  1.89097275 ,-0.68969743 ,-0.69849664, -0.6540787,
                        -2.07049155 , 0.75823699,  0.11975308 , 0.12594642 ,-0.67561681 ,-0.33026324,
                        0.86826558, -0.45040382, -0.31106317 , 0.92139191,  1.5882401 , -0.01931782,
                        -0.74068016,  0.15262985,  0.61933969, -1.25173629 ,-0.06990096 , 1.40962036,
                        0.47323309 ,-1.40968015 ,-1.18388217 , 1.93881627, -0.35910843 , 0.33075125,
                        0.02806573 , 0.08024676 ,-1.28481063 , 0.07152657 ,-1.16128504 , 1.290264,
                        0.60666654 , 1.13796111 , 1.28576911, -0.13773673, -0.45522121 , 0.13978074,
                        0.31452089 , 0.65256346 , 0.73105478, -0.8327662 ,  0.41297878 ,-1.85392176,
                        0.24883293]) # Initial weights found by Adamm optimizer + 10 inner iterations
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
    method = 'BFGS'
    result = opt.minimize(f, weights, jac=df, method=method,
                                              options={'maxiter': epochs, 'gtol': 1.e-100}, 
                                              callback=callback)
    weights = result.x
    print('Minimzed Loss', f(weights), df(weights))
    print('Minimization Result', result)

    # Post-processing
    x_axis = np.arange(len(losses))
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.semilogy(x_axis, losses, label='Training Loss')
    plt.semilogy(x_axis, grad_norms, label='Loss Gradient')
    plt.xlabel('Epoch')
    plt.title(method)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    trainNKNetBFGS()