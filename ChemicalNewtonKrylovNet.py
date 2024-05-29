import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from autograd import jacobian

import api.NewtonKrylovRecursiveNet as recnet
import api.Scheduler as sch
import api.algorithms.Adam as adam

import Deterministic_PDE as pde

def setupNeuralNetwork(outer_iterations=3, inner_iterations=4, baseweight=4.0):
    # Define the Objective Function Psi
    d2 = 0.06
    M = 200
    T = 5.e-3 # 5dt
    def psi(x):
        xp = pde.PDE_Timestepper(x, T, M, d2, verbose=False) # Use PDE first
        return x - xp
    
    # Sample Random Initial Conditions
    N_data = 1000
    rng = rd.RandomState()
    directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
    filename = 'Steady_State_LBM_dt=1e-4.npy'
    x0 = np.load(directory + filename).flatten()
    x0_data = np.array([x0,]*N_data).transpose() + rng.normal(0.0, 1.0, size=(2*M, N_data))

    # Setup classes for training
    net = recnet.NewtonKrylovSuperStructure(psi, x0_data, outer_iterations, inner_iterations, baseweight=baseweight)
    f = lambda w: net.loss(w)
    df = jacobian(f)

    return net, f, df

def sampleWeights(net):
    rng = rd.RandomState()
    inner_iterations = net.inner_iterations
    n_weights = (inner_iterations * (inner_iterations + 1) ) // 2

    weights = 0.0*rng.normal(size=n_weights)
    return weights

def trainNKNetBFGS():
    net, f, df = setupNeuralNetwork(outer_iterations=2, inner_iterations=4)
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
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.semilogy(x_axis, losses, label='Training Loss')
    plt.semilogy(x_axis, grad_norms, label='Loss Gradient')
    plt.xlabel('Epoch')
    plt.title(method)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    trainNKNetBFGS()