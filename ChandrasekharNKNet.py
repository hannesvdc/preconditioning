import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.optimize.nonlin as nl

from autograd import jacobian

import api.FastNewtonKrylovNeuralNet as nknet
import api.Scheduler as sch
import api.algorithms.Adam as adam

# General setup routine shared by all training routines
def setupRecNet(outer_iterations=3, inner_iterations=4, baseweight=4.0):
    # Define the Chandresakhar H-function
    c = 0.875
    m = 10
    mu = (np.arange(1, m+1, 1) - 0.5) / m
    def computeAc():
        Ac = np.zeros((10,10))
        for i in range(10):
            for j in range(10):
                Ac[i,j] = mu[i] / (mu[i] + mu[j])
        return 0.5 * c/m * Ac
    Ac = computeAc()
    H = lambda x: x + 1.0 / (1.0 + np.dot(Ac, x))

    # Sample data - the inittial conditions x_0,i, i = data index
    N_data = 1000
    rng = rd.RandomState()
    x0_data = rng.normal(1.0, np.sqrt(0.2), size=(m, N_data))

    # Setup classes for training
    net = nknet.NewtonKrylovNetwork(H, outer_iterations, inner_iterations, baseweight=baseweight)
    loss_fn = lambda w: net.loss(x0_data, w)
    d_loss_fn = jacobian(loss_fn)

    return net, loss_fn, d_loss_fn, H

def sampleWeights(net, loss_fn, threshold=1.e6):
    rng = rd.RandomState()
    inner_iterations = net.inner_iterations
    n_weights = (inner_iterations * (inner_iterations + 1) ) // 2

    print('Selecting Proper Random Initial Condition')
    while True:
        weights = rng.normal(size=n_weights)
        if loss_fn(weights) < threshold:
            return weights

# Only used to train Newton-Krylov network with 10 inner iterations
def trainNKNetAdam():
    net, loss_fn, d_loss_fn, H = setupRecNet(outer_iterations=3, inner_iterations=4)
    weights = sampleWeights(net, loss_fn, threshold=1.e3)
    print('Initial Loss', loss_fn(weights))
    print('Initial Loss Derivative', lg.norm(d_loss_fn(weights)))

    # Setup the optimizer
    scheduler = sch.PiecewiseConstantScheduler({0: 1.e-2, 1000: 1.e-3, 2500: 1.e-4})
    optimizer = adam.AdamOptimizer(loss_fn, d_loss_fn, scheduler=scheduler)
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
    plt.semilogy(x_axis, grad_norms, label='Loss Gradient', alpha=0.7)
    plt.xlabel('Epoch')
    plt.title('Adam')
    plt.legend()
    plt.show()

    return weights

def testNKNet(weights=None):
    # Setup the network and load the weights. All training done using BFGS routine above.
    net, _, _, H = setupRecNet(outer_iterations=3, inner_iterations=4)
    if weights is None:
        weights = np.array([-1.84575363,  0.70383152,  1.22956814, -0.08314529,  1.01171562,  0.66138989,
                            -1.52606657, -1.22742543, -0.86158424,  0.32183093])

    # Generate test data. Same distribution as training data. Test actual training data next
    m = 10
    N_data = 1000
    rng = rd.RandomState()
    x0_data = rng.normal(1.0, np.sqrt(0.2), size=(m, N_data))

    # Run each rhs through the neural network
    n_outer_iterations = 10 # Does not need be the same as the number the network was trained on.
    errors_4     = np.zeros((N_data, n_outer_iterations+1))
    nk_errors_4  = np.zeros((N_data, n_outer_iterations+1))

    samples_4  = net.forward(x0_data, weights, n_outer_iterations)
    for n in range(N_data):
        x0 = x0_data[:,n]

        for k in range(samples_4.shape[1]):
            err = lg.norm(H(samples_4[:,k,n]))
            errors_4[n,k] = err

        for k in range(n_outer_iterations+1):
            try:
                x_out = opt.newton_krylov(H,  x0, rdiff=1.e-8, iter=k, maxiter=k, method='gmres', inner_maxiter=1, outer_k=0, line_search=None)
            except nl.NoConvergence as e:
                x_out = e.args[0]
            nk_errors_4[n,k] = lg.norm(H(x_out))

    # Average the errors
    avg_errors_4  = np.average(errors_4, axis=0)
    avg_nk_errors_4  = np.average(nk_errors_4, axis=0)

    # Plot the errors
    fig, ax = plt.subplots()  
    k_axis = np.linspace(0, n_outer_iterations, n_outer_iterations+1)
    rect = mpl.patches.Rectangle((net.outer_iterations+0.5, 1.e-8), 7.5, 70, color='gray', alpha=0.2)
    ax.add_patch(rect)
    plt.semilogy(k_axis, avg_errors_4, label=r'Newton-Krylov Network with $4$ Inner Iterations', linestyle='--', marker='d')
    plt.semilogy(k_axis, avg_nk_errors_4, label=r'Scipy with $4$ Krylov Vectors', linestyle='--', marker='d')
    plt.xticks(np.linspace(0, n_outer_iterations, n_outer_iterations+1))
    plt.xlabel(r'# Outer Iterations $k$')
    plt.ylabel(r'$|H(x_k)|$')
    plt.xlim((-0.5,n_outer_iterations + 0.5))
    plt.ylim((0.1*min(np.min(avg_nk_errors_4), np.min(avg_errors_4)),70))
    plt.title(r'Function Value $|H(x_k)|$')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    weights = trainNKNetAdam()
    testNKNet(weights)