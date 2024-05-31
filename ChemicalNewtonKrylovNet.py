import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import scipy.optimize as opt
import matplotlib as mpl
import matplotlib.pyplot as plt

from autograd import jacobian

import api.NewtonKrylovRecursiveNet as recnet
import Deterministic_PDE as pde

def setupNeuralNetwork(outer_iterations=3, inner_iterations=4, baseweight=4.0):
    # Define the Objective Function Psi
    parameters = {'d2': 0.06, 'M': 200, 'T': 5.e-4}
    def psi(x):
        xp = pde.PDE_Timestepper(x, parameters['T'], parameters['M'], parameters['d2'], verbose=False) # Use PDE first
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

    return net, f, df, x0_data, parameters

def sampleWeights(net):
    rng = rd.RandomState()
    inner_iterations = net.inner_iterations
    n_weights = (inner_iterations * (inner_iterations + 1) ) // 2

    weights = 0.0*rng.normal(size=n_weights)
    return weights

def trainNKNetBFGS():
    net, f, df, _, parameters = setupNeuralNetwork(outer_iterations=2, inner_iterations=10)
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
    plt.suptitle('Chemical Reaction Newton-Krylov Neural Network')
    plt.title(r'Inner Iterations = ' + str(net.inner_iterations) + r',  $T$ = ' + str(parameters['T']))
    plt.legend()
    plt.show()

def testNewtonKrylovNet():
    # Setup the network, weights obtained by BFGS training (subroutine above)
    net, _,  _, x0_data, parameters = setupNeuralNetwork(outer_iterations=2, inner_iterations=4)
    weights = np.array([-1.919e+00, -2.155e+00, -1.756e+00, -2.206e+00, -1.996e+00,
                        -1.641e+00,  2.354e+00,  2.453e+00,  2.566e+00,  2.265e+00,])
    M = parameters['M']
    
    # Run all data througgh the neural network
    N_data = x0_data.shape[1]
    n_outer_iterations = 10
    errors = np.zeros((N_data, n_outer_iterations+1))
    for n in range(N_data):
        print('n =', n)
        x0 = x0_data[:,n]
        samples = net.forward(x0, weights, n_outer_iterations)

        for k in range(len(samples)):
            err = lg.norm(net.f(samples[k]))
            errors[n,k] = err

    # Average the errors
    avg_errors = np.average(errors, axis=0)

    # Plot the errors
    fig, ax = plt.subplots()  
    k_axis = np.linspace(0, n_outer_iterations, n_outer_iterations+1)
    rect = mpl.patches.Rectangle((net.outer_iterations+0.5, 1.e-8), n_outer_iterations-net.outer_iterations, 70, color='gray', alpha=0.2)
    ax.add_patch(rect)
    plt.semilogy(k_axis, avg_errors, label=r'$|f(x_k)|$', linestyle='--', marker='d')
    plt.xticks(np.linspace(0, n_outer_iterations, n_outer_iterations+1))
    plt.xlabel(r'# Outer Iterations $k$')
    plt.ylabel('Error')
    plt.xlim((-0.5,n_outer_iterations + 0.5))
    plt.ylim((0.1*min(np.min(avg_errors), np.min(avg_errors)),70))
    plt.suptitle('Chemical Reaction Newton-Krylov Neural Network')
    plt.title(r'Inner Iterations = ' + str(net.inner_iterations) + r',  $T$s = ' + str(parameters['T']))
    plt.legend()

    # Plot the computed steady-state solution
    index = np.argwhere(errors == np.min(errors))[0]
    print('index=', index)
    data_index = index[0]
    outer_index = index[1]
    data_point = x0_data[:, data_index]
    samples = net.forward(data_point, weights, n_outer_iterations)
    x_ss = samples[outer_index]
    U = x_ss[0:M]; V = x_ss[M:]

    x_array = np.linspace(0.0, 1.0, M)
    plt.figure()
    plt.plot(x_array, U, label=r'$U(x)$', color='red')
    plt.plot(x_array, V, label=r'$V(x)$', color='blue')
    plt.xlabel(r'$x$')
    plt.suptitle('Steady-State Newton-Krylov Neural Network')
    plt.title(r'Inner Iterations = ' + str(net.inner_iterations) + r',  $T$ = ' + str(parameters['T']))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    trainNKNetBFGS()
    #testNewtonKrylovNet()