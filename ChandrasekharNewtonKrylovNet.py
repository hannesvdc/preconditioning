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

def trainNKNetBFGS():
    inner_iterations = 10
    net, f, df = setupRecNet(outer_iterations=3, inner_iterations=inner_iterations)
    weights = sampleWeights(net, threshold=1.e4)
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

def testNKNet():
    # Setup the network and load the weights. All training done using BFGS routine above.
    net_4, _, _ = setupRecNet(outer_iterations=3, inner_iterations=4)
    weights_4 = np.array([  -1.37418947, -0.14445959,   2.51583727, -17.92557257, -112.11663472,
                               44.1241637, -4.88042119,  -50.64821349, 18.05257709, -0.54452882]) 
    net_10, _, _ = setupRecNet(outer_iterations=3, inner_iterations=10)
    weights_10 = np.array([ 3.62171888, -0.05105312,  2.47344573, -2.66141006, -1.60592837,  2.90585088,
                           -1.17764855,  3.74725482, -0.82672264, -0.31907444,  2.94074872, -1.7477809,
                            1.00887055, -1.01675493, -1.59855039,  1.72257513, -0.809649,   -2.37069611,
                            4.60295604, -1.60238219,  3.02176158, -1.79727951,  0.49863182,  0.39308403,
                           -0.41880952,  0.90450579,  1.27324059,  0.83236779,  0.52363986, -1.1758039,
                            2.41550295,  1.71376175, -0.60731314, -1.68612676,  0.64503314,  6.87339464,
                           -1.49399387, -0.89535193,  0.82095439, -0.26823959, -1.42724306,  0.10659001,
                           -0.00688416, -1.2430089,   1.55600556,  1.18506993,  0.74220142, -0.91387998,
                            2.81471078,  3.73218662,  2.23457249,  1.62909791,  3.01399171, -1.59179524,
                            0.63816011])


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
            err = lg.norm(net_4.f(samples_4[k]))
            errors_4[n,k] = err

        for k in range(len(samples_10)):
            err = lg.norm(net_10.f(samples_10[k]))
            errors_10[n,k] = err

        for k in range(n_outer_iterations+1):
            try:
                x_out = opt.newton_krylov(net_4.F, x0, rdiff=1.e-8, iter=k, method='gmres', inner_maxiter=1, outer_k=4, maxiter=k)
            except nl.NoConvergence as e:
                x_out = e.args[0]
            nk_errors_4[n,k] = lg.norm(net_4.F(x_out))

        for k in range(n_outer_iterations+1):
            try:
                x_out = opt.newton_krylov(net_10.F, x0, rdiff=1.e-8, iter=k, method='gmres', inner_maxiter=1, outer_k=10, maxiter=k)
            except nl.NoConvergence as e:
                x_out = e.args[0]
            nk_errors_10[n,k] = lg.norm(net_10.F(x_out))

    # Average the errors
    avg_errors_4  = np.average(errors_4, axis=0)
    avg_errors_10 = np.average(errors_10, axis=0)
    avg_nk_errors_4  = np.average(nk_errors_4, axis=0)
    avg_nk_errors_10 = np.average(nk_errors_10, axis=0)

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
    trainNKNetBFGS()
    #testNKNet()