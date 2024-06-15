import autograd.numpy as np
import autograd.numpy.random as rd
import autograd.numpy.linalg as lg
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.optimize.nonlin as nl

from autograd import jacobian

import api.FastNewtonKrylovNeuralNet as nknet
import api.Scheduler as sch
import api.algorithms.Adam as adam

N = 10
dx = 1.0 / N
D = 0.1

left_bc = 1.0
right_bc = -1.0

def pde_timestepper(u, f, dt):
    v = np.hstack([left_bc, u, right_bc])
    v_left = np.roll(v, 1)
    v_right = np.roll(v, -1)

    dvdt  = v
    flux = 0.5 * dt/dx * (f(v_right) - f(v_left))
    diff = 0.5 * D * dt / dx**2 * (v_right - 2.0*v + v_left)

    return (dvdt - flux + diff)[1:N]

def pde_rhs(u):
    v = np.hstack([left_bc, u, right_bc])
    v_left = np.roll(v, 1)
    v_right = np.roll(v, -1)

    flux = 0.5 * (0.5*v_right**2 - 0.5*v_left**2) / dx
    diff = 0.5 * D * (v_right - 2.0*v + v_left) / dx**2
    return (-flux + diff)[1:N]

def solvePDE():
    f = lambda u: 0.5 * np.square(u)
    rng = rd.RandomState()
    m = np.linspace(left_bc, right_bc, N+1)
    u0 = rng.normal(m[1:N], scale=0.2, size=N-1)

    dt = 1.e-5
    T = 100.0
    u = np.copy(u0)
    for n in range(int(T/dt)):
        if n % 1000 == 0:
            print('t =', (n+1)*dt)
        u = pde_timestepper(u, f, dt)

    # Compute the steady-state with Newton's method
    F = pde_rhs
    dF = jacobian(F)
    solution = opt.root(F, u0, jac=dF, tol=1.e-14, method='lm')
    u_ss = solution.x
    print('Fcn value', lg.norm(F(u_ss)), solution.success)
    print('Fcn value', lg.norm(F(u)))

    # Plot the steady state
    v_pde = np.hstack([left_bc, u, right_bc])
    v_nt = np.hstack([left_bc, u_ss, right_bc])
    x_axis = np.arange(v_pde.size) / N
    plt.plot(x_axis, v_pde, label='Steady-State by Timestepping')
    plt.plot(x_axis, v_nt, label='Steady-State by Root Finding')
    plt.xlabel(r'$x$')
    plt.legend()
    plt.show()


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

def sampleWeights(net, **kwargs):
    inner_iterations = net.inner_iterations
    n_weights = (inner_iterations * (inner_iterations + 1) ) // 2
    return np.zeros(n_weights)
        
# Only used to train Newton-Krylov network with 10 inner iterations
def trainNKNetAdam():
    net, loss_fn, d_loss_fn, _ = setupRecNet(outer_iterations=3, inner_iterations=4)
    weights = sampleWeights(net, loss_fn=loss_fn, threshold=1.e10)
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
    net, _, _, F = setupRecNet(outer_iterations=3, inner_iterations=4)

    # Generate test data. Same distribution as training data. Test actual training data next
    N_data = 1000
    rng = rd.RandomState()
    x0_data = rng.normal(np.linspace(left_bc, right_bc, N+1)[1:N,np.newaxis], scale=0.2, size=(N-1, N_data))

    # Run each rhs through the neural network
    n_outer_iterations = 10 # Does not need be the same as the number the network was trained on.
    errors    = np.zeros((N_data, n_outer_iterations+1))
    nk_errors = np.zeros((N_data, n_outer_iterations+1))

    samples  = net.forward(x0_data, weights, n_outer_iterations)
    for n in range(N_data):
        x0 = x0_data[:,n]
        for k in range(samples.shape[1]):
            err = lg.norm(pde_rhs(samples[:,k,n]))
            errors[n,k] = err

        for k in range(n_outer_iterations+1):
            try:
                x_out = opt.newton_krylov(pde_rhs, x0, rdiff=1.e-8, iter=k, maxiter=k, method='gmres', inner_maxiter=1, outer_k=0, line_search=None)
            except nl.NoConvergence as e:
                x_out = e.args[0]
            nk_errors[n,k] = lg.norm(pde_rhs(x_out))

    # Average the errors
    avg_errors = np.average(errors, axis=0)
    avg_nk_errors = np.average(nk_errors, axis=0)

    # Plot the errors
    fig, ax = plt.subplots()  
    k_axis = np.linspace(0, n_outer_iterations, n_outer_iterations+1)
    #rect = mpl.patches.Rectangle((net.outer_iterations+0.5, 1.e-8), 7.5, 70, color='gray', alpha=0.2)
    #ax.add_patch(rect)
    plt.semilogy(k_axis, avg_errors, label=r'Newton-Krylov Neural Net with $4$ Inner Iterations', linestyle='--', marker='d')
    plt.semilogy(k_axis, avg_nk_errors, label=r'Scipy newton_krylov with $4$ Krylov Vectors', linestyle='--', marker='d')
    plt.xticks(np.linspace(0, n_outer_iterations, n_outer_iterations+1))
    plt.xlabel(r'# Outer Iterations $k$')
    plt.ylabel(r'$|F(x_k)|$')
    plt.xlim((-0.5,n_outer_iterations + 0.5))
    plt.ylim((0.1*min(np.min(avg_errors), np.min(avg_nk_errors)),70))
    plt.title(r'Function Value $|F(x_k)|$')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    weights = trainNKNetAdam()
    testNKNet(weights)
