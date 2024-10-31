import autograd.numpy as np
import autograd.numpy.random as rd
import autograd.numpy.linalg as lg
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as opt
import scipy.optimize.nonlin as nl

from autograd import jacobian
import cv2
import os

import api.NewtonKrylovNeuralNet as nknet
import api.Scheduler as sch
import api.algorithms.Adam as adam

N = 100
dx = 1.0 / N
D = 0.1

left_bc = 5.0
right_bc = 1.0

def pde_timestepper(u, f, dt, left_bc_burg, right_bc_burg):
    v = np.hstack([left_bc_burg, u, right_bc_burg])
    v_left = np.roll(v, 1)
    v_right = np.roll(v, -1)

    dvdt  = v
    flux = 0.5 * dt/dx * (f(v_right) - f(v_left))
    diff = 0.5 * D * dt / dx**2 * (v_right - 2.0*v + v_left)

    return (dvdt - flux + diff)[1:N]

# def pde_rhs(u):
#     v = np.hstack([left_bc, u, right_bc])
#     v_left = np.roll(v, 1)
#     v_right = np.roll(v, -1)

#     flux = 0.5 * (0.5*v_right**2 - 0.5*v_left**2) / dx
#     diff = 0.5 * D * (v_right - 2.0*v + v_left) / dx**2
#     return (-flux + diff)[1:N]


def heat_robin_timestepper(v, dt):
    v_left = np.roll(v, 1)
    v_right = np.roll(v, -1)
    diff = 0.5 * D * (v_right - 2.0*v + v_left) / dx**2
    v_new = v + dt * diff

    # Fix robin bc's
    v_new[0] = -(D/dx) / (2 - D/dx) * v_new[1]
    v_new[N] = (D/dx) / (-2 + D/dx) * v_new[N-1]
    return v_new / np.max(v_new)

def solveRobinHeatPDE():
    sigma = 0.1
    m = np.linspace(0.0, 1.0, N+1)
    v0 = np.exp(-(m - 0.5)**2 / (2.0*sigma**2)) 
    q0 = coleHopf(v0)
    f = lambda u: 0.5 * np.square(u)
    left_bc_burg = 2.2315
    right_bc_burg = -2.2315

    dt = 1.e-4
    T = 1.0
    v = np.copy(v0)
    q = np.copy(q0)
    v_transient = [v]
    phi_transient = [np.copy(q)]
    q_transient = [np.copy(q)]
    for n in range(int(T/dt)):
        if n % 1000 == 0:
            print('t =', (n+1)*dt)
        v = heat_robin_timestepper(v, dt)
        phi = coleHopf(v)
        q = pde_timestepper(q[1:N], f, dt, left_bc_burg, right_bc_burg)
        q = np.hstack([left_bc_burg, q, right_bc_burg])

        if n % 100 == 0:
            v_transient.append(np.copy(v))
            phi_transient.append(np.copy(phi))
            q_transient.append(np.copy(q))

    image_folder = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/burgers/'
    for n in range(len(v_transient)):
        # Plot the steady state
        fig, (ax1, ax2) = plt.subplots(1, 2)
        x_axis = np.arange(v_transient[n].size) / N
        ax1.plot(x_axis, v_transient[n], label=r'$w(x, t)$', color='black')
        ax1.set_title('Heat Equation')
        ax1.set_xlabel(r'$x$')
        ax1.legend()
        ax2.plot(x_axis, phi_transient[n], label=r'$\phi(w(x, t))$', color='black')
        ax2.plot(x_axis, q_transient[n], label=r'$u(x, t)$', color='red', linestyle='dashed')
        ax2.set_title('Cole-Hopf Transform')
        ax2.set_xlabel(r'$x$')
        ax2.legend()

        plt.savefig(image_folder + str(n) + '_img.png')
        (width, height) = fig.canvas.get_width_height()
        plt.close()

    video_folder = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/burgers/'
    video_name = 'ColeHopf.avi'
    fps = 10
    video = cv2.VideoWriter(video_folder + video_name, 0, fps, (width, height))
    for n in range(len(v_transient)):
        video.write(cv2.imread(os.path.join(image_folder, str(n) + '_img.png')))
    cv2.destroyAllWindows()
    video.release()

    # Plot the steady state
    fig, (ax1, ax2) = plt.subplots(1, 2)
    x_axis = np.arange(v_transient[-1].size) / N
    ax1.plot(x_axis, v_transient[-1], label=r'$w(x, t)$')
    ax1.plot(x_axis, v0, label=r'$w(x, 0)$')
    ax1.set_title('Heat Equation')
    ax1.set_xlabel(r'$x$')
    ax1.legend()
    ax2.plot(x_axis, phi_transient[-1], label=r'$\phi(x, t)$')
    ax2.set_title('Cole-Hopf Transform')
    ax2.set_xlabel(r'$x$')
    ax2.legend()
    plt.show()

def fd(phi, dx):
    M = len(phi)
    dphi_left = (phi[1] - phi[0]) / dx
    dphi_right = (phi[-1] - phi[-2]) / dx
    dphi = np.zeros_like(phi)
    dphi[0] = dphi_left
    dphi[-1] = dphi_right

    for n in range(1, M-1):
        dphi[n] = (phi[n+1] - phi[n-1]) / (2.0 * dx)

    return dphi

def coleHopf(u):
    return fd(-D * np.log(u), dx)

def solveBurgersPDE(plot=True):
    f = lambda u: 0.5 * np.square(u)
    m = np.linspace(0.0, 1.0, N+1)
    v0 = 2.0 - 4.0*m
    u0 = v0[1:N]
    left_bc_burg = 2.2315
    right_bc_burg = -2.2315

    dt = 1.e-3
    T = 30.0
    u = np.copy(u0)
    burger_transient = [v0]
    for n in range(int(T/dt)):
        if n % 1000 == 0:
            print('t =', (n+1)*dt)
        u = pde_timestepper(u, f, dt, left_bc_burg, right_bc_burg)
        v = np.hstack([left_bc_burg, u, right_bc_burg])
        burger_transient.append(v)

    # Plot the steady state
    v_pde = np.hstack([left_bc_burg, u, right_bc_burg])
    if plot:
        x_axis = np.arange(v_pde.size) / N
        plt.plot(x_axis, v, label='Burger\'s Equation')
        plt.xlabel(r'$x$')
        plt.legend()
        plt.show()
    else:
        return v_pde


# General setup routine shared by all training routines
def setupRecNet(outer_iterations=3, inner_iterations=4, baseweight=4.0):

    def F(u): # u is a vector with 9 components
        v = np.vstack([left_bc * np.ones(u.shape[1]), u, right_bc * np.ones(u.shape[1])])
        v_left  = np.roll(v,  1, axis=0)
        v_right = np.roll(v, -1, axis=0)

        flux = 0.5 * (0.5*v_right**2 - 0.5*v_left**2) / dx
        diff = 0.5 * D * (v_right - 2.0*v + v_left) / dx**2
        return (-flux + diff)[1:N,:]

    # Sample data - the inittial conditions x_0,i, i = data index
    N_data = 1000
    rng = rd.RandomState()
    x0_data = rng.normal(np.linspace(left_bc, right_bc, N+1)[1:N,np.newaxis], scale=0.2, size=(N-1, N_data))

    # Setup classes for training
    net = nknet.NewtonKrylovNetwork(F, outer_iterations, inner_iterations, baseweight=baseweight)
    loss_fn = lambda w: net.loss(x0_data, w)
    d_loss_fn = jacobian(loss_fn)

    return net, loss_fn, d_loss_fn, F

def sampleWeights(net):
    inner_iterations = net.inner_iterations
    n_weights = (inner_iterations * (inner_iterations + 1) ) // 2
    return np.zeros(n_weights)
        
# Only used to train Newton-Krylov network with 10 inner iterations
def trainNKNetAdam():
    net, loss_fn, d_loss_fn, _ = setupRecNet(outer_iterations=3, inner_iterations=4)
    weights = sampleWeights(net)
    print('Initial Loss', loss_fn(weights))
    print('Initial Loss Derivative', lg.norm(d_loss_fn(weights)))

    # Setup the optimizer
    scheduler = sch.PiecewiseConstantScheduler({0: 1.e-2, 2500: 1.e-3})
    optimizer = adam.AdamOptimizer(loss_fn, d_loss_fn, scheduler=scheduler)
    print('Initial weights', weights)

    # Do the training
    epochs = 3000
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
    # net, _, _, F = setupRecNet(outer_iterations=3, inner_iterations=4)

    # # Generate test data. Same distribution as training data. Test actual training data next
    # N_data = 1000
    # rng = rd.RandomState()
    # x0_data = rng.normal(np.linspace(left_bc, right_bc, N+1)[1:N,np.newaxis], scale=0.2, size=(N-1, N_data))

    # # Run each rhs through the neural network
    # n_outer_iterations = 10 # Does not need be the same as the number the network was trained on.
    # errors    = np.zeros((N_data, n_outer_iterations+1))
    # nk_errors = np.zeros((N_data, n_outer_iterations+1))

    # samples  = net.forward(x0_data, weights, n_outer_iterations)
    # for n in range(N_data):
    #     x0 = x0_data[:,n]
    #     for k in range(samples.shape[1]):
    #         err = lg.norm(pde_rhs(samples[:,k,n]))
    #         errors[n,k] = err

    #     for k in range(n_outer_iterations+1):
    #         try:
    #             x_out = opt.newton_krylov(pde_rhs, x0, rdiff=1.e-8, iter=k, maxiter=k, method='gmres', inner_maxiter=1, outer_k=0, line_search=None)
    #         except nl.NoConvergence as e:
    #             x_out = e.args[0]
    #         nk_errors[n,k] = lg.norm(pde_rhs(x_out))

    # # Average the errors
    # avg_errors = np.average(errors, axis=0)
    # avg_nk_errors = np.average(nk_errors, axis=0)

    # # Plot the errors
    # fig, ax = plt.subplots()  
    # k_axis = np.linspace(0, n_outer_iterations, n_outer_iterations+1)
    # rect = mpl.patches.Rectangle((net.outer_iterations+0.5, 1.e-8), 7.5, 70, color='gray', alpha=0.2)
    # ax.add_patch(rect)
    # plt.semilogy(k_axis, avg_errors, label=r'Newton-Krylov Neural Net with $4$ Inner Iterations', linestyle='--', marker='d')
    # plt.semilogy(k_axis, avg_nk_errors, label=r'Scipy newton_krylov with $4$ Krylov Vectors', linestyle='--', marker='d')
    # plt.xticks(np.linspace(0, n_outer_iterations, n_outer_iterations+1))
    # plt.xlabel(r'# Outer Iterations $k$')
    # plt.ylabel(r'$|F(x_k)|$')
    # plt.xlim((-0.5,n_outer_iterations + 0.5))
    # plt.ylim((0.1*min(np.min(avg_errors), np.min(avg_nk_errors)),70))
    # plt.title(r'Function Value $|F(x_k)|$')
    # plt.legend()
    # plt.show()

    # Show the found steady-state solution versus the exact solution to the PDE
    v_pde = solveBurgersPDE(plot=False)
    #nknet = np.hstack([left_bc, samples[:,10,0], right_bc])
    x_array = np.arange(v_pde.size) / N
    plt.plot(x_array, v_pde, label='Exact Solution')
    #plt.plot(x_array, nknet, label='Newton-Krylov Net Solution')
    plt.legend()
    plt.title('Viscous-Burgers Equation')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u(x)$', rotation=0)
    plt.show()

if __name__ == '__main__':
    #weights = trainNKNetAdam()
    solveRobinHeatPDE()
    #solvePDE()
