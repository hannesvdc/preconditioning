import autograd.numpy as np
import autograd.numpy.random as rd
import autograd.numpy.linalg as lg
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as opt
import scipy.optimize.nonlin as nl

from autograd import grad, jacobian

import api.NewtonKrylovNeuralNet as nknet
import api.Scheduler as sch
import api.algorithms.Adam as adam

def MB(x):
    return -200.0 * np.exp(-(x[0] - 1.0)**2 - 10.0*x[1]**2)\
           -100.0 * np.exp(-x[0]**2 - 10.0*(x[1] - 0.5)**2)\
           -170.0 * np.exp(-6.5*(x[0] + 0.5)**2 + 11.0*(x[0] + 0.5)*(x[1] - 1.5) - 6.5*(x[1] - 1.5)**2)\
            +15.0 * np.exp( 0.7*(x[0] + 1.0)**2 +  0.6*(x[0] + 1.0)*(x[1] - 1.0) + 0.7*(x[1] - 1.0)**2)

def dMB(x):
    t1 = -200.0 * np.exp(-(x[0] - 1.0)**2 - 10.0*x[1]**2)
    t2 = -100.0 * np.exp(-x[0]**2 - 10.0*(x[1] - 0.5)**2)
    t3 = -170.0 * np.exp(-6.5*(x[0] + 0.5)**2 + 11.0*(x[0] + 0.5)*(x[1] - 1.5) - 6.5*(x[1] - 1.5)**2)
    t4 =   15.0 * np.exp( 0.7*(x[0] + 1.0)**2 +  0.6*(x[0] + 1.0)*(x[1] - 1.0) + 0.7*(x[1] - 1.0)**2)

    MB_x = -2.0 * (x[0] - 1.0) * t1 \
           -2.0 * x[0] * t2 +\
           (-2.0 * 6.5 * (x[0] + 0.5) + 11.0 * (x[1] - 1.5)) * t3 +\
           ( 2.0 * 0.7*(x[0] + 1.0) + 0.6 * (x[1] - 1.0)) * t4
    MB_y = -2.0 * 10.0 * x[1] * t1 \
           -2.0 * 10.0 * (x[1] - 0.5) * t2 +\
           (11.0 * (x[0] + 0.5) - 2.0 * 6.5 * (x[1] - 1.5)) * t3 +\
           ( 0.6 * (x[0] + 1.0) + 2.0 * 0.7 * (x[1] - 1.0)) * t4
    return np.vstack((MB_x, MB_y))

def sampleInitials(N_data):
    E = -20.0
    rng = rd.RandomState()
    samples = np.zeros((2, N_data))

    index = 0
    while index < N_data:
        x = rng.uniform(low=-1.50, high=1.0)
        y = rng.uniform(low=-0.25, high=1.75)
        z = np.array([x, y])

        if MB(z) <= E:
            samples[:,index] = z
            index += 1

    return samples

def setupNKNet(outer_iterations=3, inner_iterations=4, baseweight=4.0):
    # Sample data - the inittial conditions x_0,i, i = data index
    N_data = 1000
    x0_data = sampleInitials(N_data)
    
    # Setup classes for training
    net = nknet.NewtonKrylovNetwork(dMB, outer_iterations, inner_iterations, baseweight=baseweight)
    loss_fn = lambda w: net.loss(x0_data, w)
    d_loss_fn = jacobian(loss_fn)

    return net, loss_fn, d_loss_fn

def sampleWeights(net):
    inner_iterations = net.inner_iterations
    n_weights = (inner_iterations * (inner_iterations + 1) ) // 2

    rng = rd.RandomState()
    weights = rng.normal(loc=0.0, scale=0.01, size=n_weights)
    return np.zeros(n_weights)

def optimizeNKNetAdam():
    inner_iterations = 7
    outer_iterations = 3
    net, loss_fn, d_loss_fn = setupNKNet(outer_iterations=outer_iterations, inner_iterations=inner_iterations)
    weights = sampleWeights(net)
    print('Initial Loss', loss_fn(weights))
    print('Initial Loss Derivative', lg.norm(d_loss_fn(weights)))

    # Setup the optimizer
    scheduler = sch.PiecewiseConstantScheduler({0: 1.e-4, 1000: 1.e-5})
    optimizer = adam.AdamOptimizer(loss_fn, d_loss_fn, scheduler=scheduler)
    print('Initial weights', weights)

    # Do the training
    epochs = 25000
    try:
        weights = optimizer.optimize(weights, n_epochs=epochs)
    except KeyboardInterrupt: # If Training has converged well enough with Adam, the user can stop manually
        print('Aborting Training. Plotting Convergence')
        weights = optimizer.lastweights
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
    net, _, _ = setupNKNet(outer_iterations=3, inner_iterations=7)

    # Generate test data. Same distribution as training data. Test actual training data next
    N_data = 1000
    data = sampleInitials(N_data)

    # Run each rhs through the neural network
    n_outer_iterations = 10 # Does not need be the same as the number the network was trained on.
    errors    = np.zeros((N_data, n_outer_iterations+1))
    nk_errors = np.zeros((N_data, n_outer_iterations+1))

    samples  = net.forward(data, weights, n_outer_iterations)
    for n in range(N_data):
        x0 = data[:,n]
        for k in range(samples.shape[1]):
            err = lg.norm(dMB(samples[:,k,n]))
            errors[n,k] = err

        for k in range(n_outer_iterations+1):
            try:
                pass
                #x_out = opt.newton_krylov(dMB, x0, rdiff=1.e-8, iter=k, maxiter=k, method='gmres', inner_maxiter=1, outer_k=0, line_search=None)
            except nl.NoConvergence as e:
                x_out = e.args[0]
            #nk_errors[n,k] = lg.norm(dMB(x_out))

    # Average the errors
    avg_errors = np.average(errors, axis=0)
    avg_nk_errors = np.average(nk_errors, axis=0)

    # Plot the errors
    fig, ax = plt.subplots()  
    k_axis = np.linspace(0, n_outer_iterations, n_outer_iterations+1)
    rect = mpl.patches.Rectangle((net.outer_iterations+0.5, 1.e-8), 7.5, 70, color='gray', alpha=0.2)
    ax.add_patch(rect)
    plt.semilogy(k_axis, avg_errors, label=r'Newton-Krylov Neural Net with $4$ Inner Iterations', linestyle='--', marker='d')
    #plt.semilogy(k_axis, avg_nk_errors, label=r'Scipy newton_krylov with $4$ Krylov Vectors', linestyle='--', marker='d')
    plt.xticks(np.linspace(0, n_outer_iterations, n_outer_iterations+1))
    plt.xlabel(r'# Outer Iterations $k$')
    plt.ylabel(r'$|F(x_k)|$')
    plt.xlim((-0.5,n_outer_iterations + 0.5))
    #plt.ylim((0.1*min(np.min(avg_errors), np.min(avg_nk_errors)),70))
    plt.title(r'Function Value $|F(x_k)|$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    weights = optimizeNKNetAdam()
    testNKNet(weights)
