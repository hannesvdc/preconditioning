import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import matplotlib as mpl
import matplotlib.pyplot as plt

from autograd import jacobian

import api.NewtonKrylovNeuralNet as nknet
import api.Scheduler as sch
import api.algorithms.Adam as adam
import Deterministic_PDE as pde

def setupNeuralNetwork(T, outer_iterations=3, inner_iterations=4, baseweight=4.0):
    # Define the Objective Function Psi
    M = 200
    d2 = 0.06
    parameters = {'d2': d2, 'M': M, 'T': T}
    def psi(x):
        xp = pde.PDE_Timestepper(x, parameters) # Use PDE first
        return x - xp # approx T * \partial_T phi_pde(x)
    
    # Sample Random Initial Conditions
    seed = 100
    N_data = 1000
    rng = rd.RandomState(seed=seed)
    directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
    filename = 'Steady_State_LBM_dt=1e-4.npy'
    x0 = np.load(directory + filename).flatten()
    x0_data = np.array([x0,]*N_data).transpose() + rng.normal(0.0, 1.0, size=(2*M, N_data))

    # Setup classes for training
    net = nknet.NewtonKrylovNetwork(psi, outer_iterations, inner_iterations, baseweight=baseweight)
    f = lambda w: net.loss(x0_data, w)
    df = jacobian(f)

    return net, f, df, x0_data, parameters, psi

def sampleWeights(loss_fn, n_inner):
    rng = rd.RandomState()
    inner_iterations = n_inner
    n_weights = (inner_iterations * (inner_iterations + 1) ) // 2

    while True:
        weights = rng.normal(scale=0.1, size=n_weights)
        l = loss_fn(weights)
        if l < 1.e4:
            return weights
        
def trainNKNetAdam(T, n_inner):
    _, loss_fn, d_loss_fn, _, parameters, _ = setupNeuralNetwork(T=T, outer_iterations=2, inner_iterations=n_inner)
    weights = sampleWeights(loss_fn, n_inner)
    print('Initial Loss', loss_fn(weights))
    print('Initial Loss Derivative', lg.norm(d_loss_fn(weights)))

    # Setup the optimizer
    scheduler = sch.PiecewiseConstantScheduler({0: 1.e-2, 500: 1.e-3})
    optimizer = adam.AdamOptimizer(loss_fn, d_loss_fn, scheduler=scheduler)
    print('Initial weights', weights)

    # Do the training
    epochs = 15000
    try:
        weights = optimizer.optimize(weights, n_epochs=epochs)
    except KeyboardInterrupt: # If Training has converged well enough with Adam, the user can stop manually
        print('Aborting Training. Plotting Convergence')
        weights = optimizer.lastweights
    print('Done Training at', len(optimizer.losses), 'epochs. Weights =', weights)
    losses = np.array(optimizer.losses)
    grad_norms = np.array(optimizer.gradient_norms)

    # Storing weights
    directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/NKNet/'
    filename = 'Weights_Adam_T=' + str(T) + '_inner_iterations=' + str(n_inner) + '_.npy'
    np.save(directory + filename, weights)

    # Post-processing
    x_axis = np.arange(len(losses))
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.semilogy(x_axis, losses, label='Training Loss')
    plt.semilogy(x_axis, grad_norms, label='Loss Gradient', alpha=0.7)
    plt.xlabel('Epoch')
    plt.suptitle('Chemical Reaction Newton-Krylov Neural Network')
    plt.title(r'Adam, Inner Iterations = ' + str(n_inner) + r',  $T$ = ' + str(parameters['T']))
    plt.legend()
    plt.show()

    return weights

def testNewtonKrylovNet(T, n_inner, weights):
    # Setup the network, weights obtained by BFGS training (subroutine above)
    net, _,  _, x0_data, parameters, psi = setupNeuralNetwork(T, outer_iterations=2, inner_iterations=n_inner)
    M = parameters['M']
    
    # Run all data through the neural network
    N_data = x0_data.shape[1]
    n_outer_iterations = 10
    samples = net.forward(x0_data, weights, n_outer_iterations)
    errors = np.zeros((N_data, n_outer_iterations+1))
    for n in range(N_data):
        for k in range(samples.shape[1]):
            err = lg.norm(psi(samples[:,k,n]))
            errors[n,k] = err

    # Average the errors
    avg_errors = np.average(errors, axis=0)

    # Plot the errors
    _, ax = plt.subplots()  
    k_axis = np.linspace(0, n_outer_iterations, n_outer_iterations+1)
    rect = mpl.patches.Rectangle((net.outer_iterations+0.5, 1.e-8), n_outer_iterations-net.outer_iterations, 70, color='gray', alpha=0.2)
    ax.add_patch(rect)
    plt.semilogy(k_axis, avg_errors, label=r'$|\psi(x_k)|$', linestyle='--', marker='d')
    plt.xticks(np.linspace(0, n_outer_iterations, n_outer_iterations+1))
    plt.xlabel(r'# Outer Iterations $k$')
    plt.ylabel('Error')
    plt.xlim((-0.5,n_outer_iterations + 0.5))
    plt.ylim((0.1*min(np.min(avg_errors), np.min(avg_errors)),70))
    plt.suptitle('Chemical Reaction Newton-Krylov Neural Network')
    plt.title(r'Inner Iterations = ' + str(net.inner_iterations) + r',  $T$s = ' + str(T))
    plt.legend()

    # Plot the computed steady-state solution
    index = np.argwhere(errors == np.min(errors))[0][0]
    x_ss = samples[:,10,index]
    U = x_ss[0:M]; V = x_ss[M:]

    x_array = np.linspace(0.0, 1.0, M)
    plt.figure()
    plt.plot(x_array, U, label=r'$U_s(x)$', color='red')
    plt.plot(x_array, V, label=r'$V_s(x)$', color='blue')
    plt.xlabel(r'$x$')
    plt.suptitle('Steady-State Newton-Krylov Neural Network')
    plt.title(r'Inner Iterations = ' + str(net.inner_iterations) + r',  $T$ = ' + str(parameters['T']))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    T = 5.e-4
    n_inner = 25
    weights = trainNKNetAdam(T, n_inner)
    testNewtonKrylovNet(T, n_inner, weights)