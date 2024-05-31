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

def trainNKNetAdam():
    net, f, df = setupRecNet(outer_iterations=3, inner_iterations=4)
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
    net, f, df = setupRecNet(outer_iterations=3, inner_iterations=10)
    #weights = np.array([-1.93181061,  0.18862427 ,-0.36436414,  1.75800653,  0.81753954 ,-2.90153424,
    #                    1.11418358 ,-1.1968051 ,  0.35490947 , 0.77058088]) # Initial weights found by Adamm optimizer for 4 inner iterations (10 weights)
    # weights = np.array([-1.24055358 ,-0.4106254  , 0.53113248 ,-2.16433248 , 0.65725721 ,-0.21630475,
    #                     -1.17453146 , 2.24363514,  1.89097275 ,-0.68969743 ,-0.69849664, -0.6540787,
    #                     -2.07049155 , 0.75823699,  0.11975308 , 0.12594642 ,-0.67561681 ,-0.33026324,
    #                      0.86826558, -0.45040382, -0.31106317 , 0.92139191,  1.5882401 , -0.01931782,
    #                     -0.74068016,  0.15262985,  0.61933969, -1.25173629 ,-0.06990096 , 1.40962036,
    #                      0.47323309 ,-1.40968015 ,-1.18388217 , 1.93881627, -0.35910843 , 0.33075125,
    #                      0.02806573 , 0.08024676 ,-1.28481063 , 0.07152657 ,-1.16128504 , 1.290264,
    #                      0.60666654 , 1.13796111 , 1.28576911, -0.13773673, -0.45522121 , 0.13978074,
    #                      0.31452089 , 0.65256346 , 0.73105478, -0.8327662 ,  0.41297878 ,-1.85392176,
    #                      0.24883293]) # Initial weights found by Adamm optimizer for 10 inner iterations (45 weights)
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
    # Setup the network and load the weights
    inner_iterations = 10
    net, _, _ = setupRecNet(outer_iterations=3, inner_iterations=inner_iterations)
    #weights = np.array([  -1.37418947 ,  -0.14445959 ,   2.51583727 , -17.92557257 ,-112.11663472,
    #                        44.1241637   , -4.88042119,  -50.64821349  , 18.05257709  , -0.54452882]) # BFGS, inner=4
    weights = np.array([ 3.62171888, -0.05105312,  2.47344573, -2.66141006, -1.60592837,  2.90585088,
                        -1.17764855,  3.74725482, -0.82672264, -0.31907444,  2.94074872, -1.7477809,
                         1.00887055, -1.01675493, -1.59855039,  1.72257513, -0.809649  , -2.37069611,
                         4.60295604, -1.60238219,  3.02176158, -1.79727951,  0.49863182,  0.39308403,
                        -0.41880952,  0.90450579,  1.27324059,  0.83236779,  0.52363986, -1.1758039,
                         2.41550295,  1.71376175, -0.60731314, -1.68612676,  0.64503314,  6.87339464,
                        -1.49399387, -0.89535193,  0.82095439, -0.26823959, -1.42724306,  0.10659001,
                        -0.00688416, -1.2430089 ,  1.55600556,  1.18506993,  0.74220142, -0.91387998,
                         2.81471078,  3.73218662,  2.23457249,  1.62909791,  3.01399171, -1.59179524,
                         0.63816011]) # inner = 10 BFGS
    # Generate test data. Same distribution as training data. Test actual training data next
    m = 10
    N_data = 1000
    rng = rd.RandomState()
    x0_data = rng.normal(1.0, np.sqrt(0.2), size=(m, N_data))

    # Run each rhs through the neural network
    n_outer_iterations = 10 # Does not need be the same as the number the network was trained on.
    errors = np.zeros((N_data, n_outer_iterations+1))
    nk_errors = np.zeros((N_data, n_outer_iterations+1))
    for n in range(N_data):
        x0 = x0_data[:,n]
        samples = net.forward(x0, weights, n_outer_iterations)

        for k in range(len(samples)):
            err = lg.norm(net.f(samples[k]))
            errors[n,k] = err

        for k in range(n_outer_iterations+1):
            try:
                x_out = opt.newton_krylov(net.f, x0, rdiff=1.e-8, iter=k, method='gmres', inner_maxiter=1, outer_k=inner_iterations, maxiter=k)
            except nl.NoConvergence as e:
                x_out = e.args[0]
            nk_errors[n,k] = lg.norm(net.f(x_out))

    # Average the errors
    avg_errors = np.average(errors, axis=0)
    min_errors = np.min(errors, axis=0)
    avg_nk_errors = np.average(nk_errors, axis=0)

    # Plot the errors
    fig, ax = plt.subplots()  
    k_axis = np.linspace(0, n_outer_iterations, n_outer_iterations+1)
    rect = mpl.patches.Rectangle((net.outer_iterations+0.5, 1.e-8), 7.5, 70, color='gray', alpha=0.2)
    ax.add_patch(rect)
    plt.semilogy(k_axis, avg_errors, label='Averaged Neural Network Output Error', linestyle='--', marker='d')
    plt.semilogy(k_axis, min_errors, label='Minimal Neural Network Output Error', linestyle='--', marker='d')
    plt.semilogy(k_axis, avg_nk_errors, label='Averaged Scipy Newton-Krylov Error', linestyle='--', marker='d')
    plt.xticks(np.linspace(0, n_outer_iterations, n_outer_iterations+1))
    plt.xlabel(r'# Outer Iterations $k$')
    plt.ylabel('Error')
    plt.xlim((-0.5,n_outer_iterations + 0.5))
    plt.ylim((0.1*min(np.min(avg_nk_errors), np.min(avg_errors)),70))
    plt.suptitle(r'Newton-Krylov Neural Net')
    plt.title('Inner Iterations = ' + str(net.inner_iterations))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #trainNKNetBFGS()
    testNKNet()