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

def is_pos_def(B):
    return np.all(lg.eigvals(B) > 0)

def generateData():
    # Sample Data
    N_data = 1000
    rng = rd.RandomState()
    b_mean = np.array([2.483570, -0.691321, 3.238442, 7.615149, -1.170766])
    b_mean_repeated = np.array([b_mean,]*N_data).transpose()
    b = b_mean_repeated + rng.uniform(low=-1, high=1, size=(5, N_data))

    A_mean = np.array([[1.392232, 0.152829, 0.088680, 0.185377, 0.156244],
                       [0.152829, 1.070883, 0.020994, 0.068940, 0.141251],
                       [0.088680, 0.020994, 0.910692,-0.222769, 0.060267],
                       [0.185377, 0.068940,-0.222769, 0.833275, 0.058072],
                       [0.156244, 0.141251, 0.060267, 0.058072, 0.735495]])
    A = np.zeros((5, 5, N_data))
    for n in range(N_data):
        M = A_mean + rng.uniform(low=-1, high=1, size=(5, 5))
        M = 0.5*(M + M.transpose()) + 5.0*np.eye(5)
        all_pos_def = (all_pos_def and is_pos_def(M))
        A[:,:,n] = np.copy(M)

    # Save the Training Data once and for all
    directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/'
    A_filename = 'A_Training_Data.npy'
    b_filename = 'b_Training_Data.npy'
    np.save(directory + A_filename, A)
    np.save(directory + b_filename, b)

# General setup routine shared by all training routines
def setupRecNet(outer_iterations=3, inner_iterations=4, baseweight=4.0):
    directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/'
    A_filename = 'A_Training_Data.npy'
    b_filename = 'b_Training_Data.npy'
    A_data = np.load(directory + A_filename)
    b_data = np.load(directory + b_filename)

    # Setup classes for training
    net = recnet.R2N2(A_data, b_data, outer_iterations, inner_iterations, baseweight=baseweight)
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
        if loss < 1.e9:
            return weights

def trainRecNetAdam():
    net, f, df = setupRecNet(outer_iterations=3, inner_iterations=4)
    weights = sampleWeights(net)
    print('Initial Loss', f(weights))
    print('Initial Loss Derivative', lg.norm(df(weights)))

    # Setup the optimizer
    scheduler = sch.PiecewiseConstantScheduler({0: 0.001, 1000: 0.001, 5000: 1.e-4, 10000: 1.e-5, 15000: 1.e-6})
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

def trainRecNetBFGS():
    net, f, df = setupRecNet(outer_iterations=3, inner_iterations=4)
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
    method = 'BFGS'
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

def trainRecNetBFGSImpl(): # Train NN with own bfgs implementation
    net, f, df = setupRecNet(outer_iterations=3, inner_iterations=4)
    weights = sampleWeights(net)
    print('Initial Loss', f(weights))
    print('Initial Loss Derivative', lg.norm(df(weights)))

    learning_rate = 1.e-4
    optimizer = bfgs.BFGSOptimizer(f, df, sch.PiecewiseConstantScheduler({0: learning_rate}))

    epochs = 5000
    tolerance = 1.e-8
    weights = optimizer.optimize(weights, maxiter=epochs, tolerance=tolerance)
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