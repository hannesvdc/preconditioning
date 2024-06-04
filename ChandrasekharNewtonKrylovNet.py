import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.optimize.nonlin as nl

from autograd import jacobian

import api.NewtonKrylovRecursiveNet as recnet
import api.Scheduler as sch
import api.algorithms.Adam as adam

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

    return net, f, df, H

def sampleWeights(net, threshold=1.e6):
    rng = rd.RandomState()
    inner_iterations = net.inner_iterations
    n_weights = (inner_iterations * (inner_iterations + 1) ) // 2

    print('Selecting Proper Random Initial Condition')
    while True:
        weights = rng.normal(size=n_weights)
        if net.loss(weights) < threshold:
            return weights

# Only used to train Newton-Krylov network with 10 inner iterations
def trainNKNetAdam():
    net, f, df, H = setupRecNet(outer_iterations=3, inner_iterations=10)
    weights = sampleWeights(net, threshold=1.e3)
    print('Initial Loss', f(weights))
    print('Initial Loss Derivative', lg.norm(df(weights)))

    # Setup the optimizer
    scheduler = sch.PiecewiseConstantScheduler({0: 1.e-3, 1000: 1.e-4, 10000: 1.e-5})
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

# Used to train Newton-Krylov network with 4 inner iterations, and to fine-tune network with 10 inner iterations.
def trainNKNetBFGS():
    inner_iterations = 10
    net, f, df, H = setupRecNet(outer_iterations=3, inner_iterations=inner_iterations)
    if inner_iterations == 4:
        weights = sampleWeights(net, threshold=1.e3)
    else:
        pass
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
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.semilogy(x_axis, losses, label='Training Loss')
    plt.semilogy(x_axis, grad_norms, label='Loss Gradient')
    plt.xlabel('Epoch')
    plt.title(method)
    plt.legend()
    plt.show()

def testNKNet():
    # Setup the network and load the weights. All training done using BFGS routine above.
    net_4, _, _, H = setupRecNet(outer_iterations=3, inner_iterations=4)
    #weights_4 = np.array([  -1.37418947, -0.14445959,   2.51583727, -17.92557257, -112.11663472,
    #                           44.1241637, -4.88042119,  -50.64821349, 18.05257709, -0.54452882])
    weights_4 = np.array([ -1.13845701,  -21.48290808, 151.35155532, -14.0296891,   82.79574916,
                            0.47406062,  -6.7392919,   45.33721111,  -1.37527638,   1.01790898])
    net_10, _, _, _ = setupRecNet(outer_iterations=3, inner_iterations=10)
    weights_10 = np.array([-1.46423232e+01,  8.37854544e+00, -2.52518182e+00, -4.61474474e+00,
                            1.42699054e+00,  2.89648642e+00,  6.90150675e-01, -2.15219208e-01,
                           -4.42460933e-01,  5.31963320e-01, -1.94328619e+00,  6.06814749e-01,
                            1.19894827e+00, -1.44633160e+00,  4.05631099e+00,  2.32673885e-01,
                           -6.84036521e-02, -1.37796996e-01,  1.69128993e-01, -4.66832905e-01,
                            4.06089839e-01,  9.18898833e-01, -2.79862752e-01, -5.57274381e-01,
                            6.77093512e-01, -1.88653307e+00,  1.63397934e+00,  4.04415056e+00,
                           -5.73551339e-02,  1.73017382e-02,  3.45569788e-02, -4.21027207e-02,
                            1.17011853e-01, -1.01467904e-01, -2.51501343e-01,  1.11177301e-01,
                            3.20706645e-02, -9.75473543e-03, -1.94369884e-02,  2.36244069e-02,
                            -6.57962093e-02, 5.69932769e-02,  1.41082050e-01, -6.24818110e-02,
                            2.32286922e-01, -3.08366828e-01,  8.48093765e-02,  1.79517894e-01,
                           -2.23825572e-01,  6.03533074e-01, -5.25335247e-01, -1.31406232e+00,
                            5.72103565e-01, -2.16376069e+00, -2.31007667e+00])

    # Generate test data. Same distribution as training data. Test actual training data next
    m = 10
    N_data = 1000
    rng = rd.RandomState()
    x0_data = rng.normal(1.0, np.sqrt(0.2), size=(m, N_data))

    # Run each rhs through the neural network
    n_outer_iterations = 10 # Does not need be the same as the number the network was trained on.
    errors_4     = np.zeros((N_data, n_outer_iterations+1))
    errors_10    = np.zeros((N_data, n_outer_iterations+1))
    nk_errors_4  = np.zeros((N_data, n_outer_iterations+1))
    nk_errors_10 = np.zeros((N_data, n_outer_iterations+1))
    for n in range(N_data):
        x0 = x0_data[:,n]
        samples_4  = net_4.forward(x0, weights_4, n_outer_iterations)
        samples_10 = net_10.forward(x0, weights_10, n_outer_iterations)

        for k in range(len(samples_4)):
            err = lg.norm(H(samples_4[k]))
            errors_4[n,k] = err

        for k in range(len(samples_10)):
            err = lg.norm(H(samples_10[k]))
            errors_10[n,k] = err

        for k in range(n_outer_iterations+1):
            try:
                x_out = opt.newton_krylov(H,  x0, rdiff=1.e-8, iter=k, method='gmres', inner_maxiter=1, inner_restart=4)
            except nl.NoConvergence as e:
                x_out = e.args[0]
            nk_errors_4[n,k] = lg.norm(H(x_out))

        for k in range(n_outer_iterations+1):
            try:
                x_out = opt.newton_krylov(H, x0, rdiff=1.e-8, iter=k, method='gmres', inner_maxiter=1, inner_restart=10)
            except nl.NoConvergence as e:
                x_out = e.args[0]
            nk_errors_10[n,k] = lg.norm(H(x_out))

    # Average the errors
    avg_errors_4  = np.average(errors_4, axis=0)
    avg_errors_10 = np.average(errors_10, axis=0)
    avg_nk_errors_4  = np.average(nk_errors_4, axis=0)
    avg_nk_errors_10 = np.average(nk_errors_10, axis=0)
    print(avg_errors_10)

    # Plot the errors
    fig, ax = plt.subplots()  
    k_axis = np.linspace(0, n_outer_iterations, n_outer_iterations+1)
    rect = mpl.patches.Rectangle((net_4.outer_iterations+0.5, 1.e-8), 7.5, 70, color='gray', alpha=0.2)
    ax.add_patch(rect)
    plt.semilogy(k_axis, avg_errors_4, label=r'Newton-Krylov Network with $4$ Inner Iterations', linestyle='--', marker='d')
    plt.semilogy(k_axis, avg_errors_10, label=r'Newton-Krylov Network with $10$ Inner Iterations', linestyle='--', marker='d')
    plt.semilogy(k_axis, avg_nk_errors_4, label=r'Scipy with $4$ Krylov Vectors', linestyle='--', marker='d')
    plt.semilogy(k_axis, avg_nk_errors_10, label=r'Scipy with $10$ Krylov Vectors', linestyle='--', marker='d')
    plt.xticks(np.linspace(0, n_outer_iterations, n_outer_iterations+1))
    plt.xlabel(r'# Outer Iterations $k$')
    plt.ylabel(r'$|H(x_k)|$')
    plt.xlim((-0.5,n_outer_iterations + 0.5))
    plt.ylim((0.1*min(np.min(avg_nk_errors_4), np.min(avg_errors_4), np.min(avg_nk_errors_10), np.min(avg_errors_10)),70))
    plt.title(r'Function Value $|H(x_k)|$')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    trainNKNetAdam()
    #trainNKNetBFGS()
    #testNKNet()