import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.sparse.linalg as slg

from autograd import jacobian

import api.KrylovRecursiveNet as recnet
import api.Scheduler as sch
import algorithms.Adam as adam


# General setup routine shared by all training routines
def setupRecNet(fixedA=True, outer_iterations=3, inner_iterations=4, baseweight=4.0):
    directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/'
    A_filename = 'A_Training_Data.npy'
    b_filename = 'b_Training_Data.npy'
    b_data = np.load(directory + b_filename)
    N_data = b_data.shape[1]

    A = np.array([[1.392232, 0.152829, 0.088680, 0.185377, 0.156244],
                  [0.152829, 1.070883, 0.020994, 0.068940, 0.141251],
                  [0.088680, 0.020994, 0.910692,-0.222769, 0.060267],
                  [0.185377, 0.068940,-0.222769, 0.833275, 0.058072],
                  [0.156244, 0.141251, 0.060267, 0.058072, 0.735495]])
    if fixedA:
        A_data = np.repeat(A[:,:,np.newaxis], N_data, axis=2)
        P = slg.spilu(A)
        P = np.matmul(P.L.toarray(), P.U.toarray())
        P_inv = lg.inv(P) # No need to cmpute inverse, but we will need P in ndarray form
    else:
        # Compute averaged matrix and preconditioner basedon that (is not A, but shifted!!!)
        A_data = np.load(directory + A_filename)
        M = np.mean(A_data, axis=2)
        assert M.shape == (5,5)
        P = slg.spilu(M)
        P = np.matmul(P.L.toarray(), P.U.toarray())
        P_inv = lg.inv(P) # No need to cmpute inverse, but we will need P in ndarray form
            
    for n in range(N_data):
        print('precond', lg.norm(np.matmul(A_data[:,:,n], P_inv), ord=2))

    # Setup classes for training
    net = recnet.KrylovSuperStructure(A_data, b_data, outer_iterations, inner_iterations, P=P_inv, baseweight=baseweight)
    f = lambda w: net.loss(w)
    df = jacobian(f)

    return net, f, df

def sampleWeights(net):
    rng = rd.RandomState()
    inner_iterations = net.inner_iterations
    n_weights = (inner_iterations * (inner_iterations + 1) ) // 2

    weights = rng.normal(size=n_weights)
    print('loss =', net.loss(weights))
    return weights
        

def trainRecNetAdam():
    net, f, df = setupRecNet(fixedA=False, outer_iterations=3, inner_iterations=4) # Change here for fixed / random A
    weights = sampleWeights(net)
    print('Initial Loss', f(weights))
    print('Initial Loss Derivative', lg.norm(df(weights)))

    # Setup the optimizer
    scheduler = sch.PiecewiseConstantScheduler({0: 1.e-3, 3000: 1.e-5, 15000: 1.e-6})
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
    net, f, df = setupRecNet(fixedA=False, outer_iterations=3, inner_iterations=4)
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
                                              options={'maxiter': epochs}, 
                                              callback=callback)
    weights = result.x
    print('Minimzed Loss', f(weights), lg.norm(df(weights)))
    print('Minimization Result', result)

    # Post-processing
    x_axis = np.arange(len(losses))
    plt.semilogy(x_axis, grad_norms, label='Gradient Norms')
    plt.semilogy(x_axis, losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.title(method)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    trainRecNetBFGS()