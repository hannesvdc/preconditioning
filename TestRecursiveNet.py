import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import matplotlib
import matplotlib.pyplot as plt

from TrainRecursiveNet import setupRecNet

def testbRecNet():
    # Setup the network and load the weights
    net, _, _ = setupRecNet()
    weights = np.array([-0.75338897, -0.69512928 ,-1.28298784, -0.8088681 ,  0.87909088 , 1.48089791,
                        -0.69293485 ,-2.75058767 ,-2.00362172 , 0.90054142]) # Adam + BFGS refinement
    
    # Generate test data. Same distribution as training data. Test actual training data next
    N_data = 1000
    rng = rd.RandomState()
    A = np.array([[1.392232, 0.152829, 0.088680, 0.185377, 0.156244],
                  [0.152829, 1.070883, 0.020994, 0.068940, 0.141251],
                  [0.088680, 0.020994, 0.910692,-0.222769, 0.060267],
                  [0.185377, 0.068940,-0.222769, 0.833275, 0.058072],
                  [0.156244, 0.141251, 0.060267, 0.058072, 0.735495]])
    b_mean = np.array([2.483570, -0.691321, 3.238442, 7.615149, -1.170766])
    b_mean_repeated = np.array([b_mean,]*N_data).transpose()
    b = b_mean_repeated + rng.uniform(low=-1, high=1, size=(5, N_data))

    # Run each rhs through the neural network
    n_outer_iterations = 10 # Does not need be the same as the number the network was trained on.
    n_inner_iterations = 4
    errors = np.zeros((N_data, n_outer_iterations+1))
    for n in range(N_data):
        rhs = b[:,n]
        samples = net.forward(weights, rhs, n_outer_iterations)

        for k in range(len(samples)):
            err = lg.norm(A.dot(samples[k]) - rhs)
            errors[n,k] = err

    # Solve the system Ax = rhs with gmres for all rhs
    class fc_eval:
        def __init__(self):
            self.func_vals = 0
        def cb(self,input):
            self.func_vals += 1

    gmres_errors = np.zeros((N_data, n_outer_iterations+1))
    for n in range(N_data):
        rhs = b[:,n]
        gmres_errors[n,0] = lg.norm(rhs)

        for k in range(1, n_outer_iterations+1):
            evaluator = fc_eval()
            x, _ = slg.gmres(A, rhs, x0=np.zeros(rhs.size), maxiter=k, restart=n_inner_iterations, tol=0.0, callback=evaluator.cb, callback_type='pr_norm')
            gmres_errors[n,k] = lg.norm(A.dot(x) - rhs)
            #print('Number of Inner Iterations', evaluator.func_vals, k)

    # Average the errors
    avg_errors = np.average(errors, axis=0)
    avg_gmres_ = np.average(gmres_errors, axis=0)

    # Plot the errors
    fig, ax = plt.subplots()  
    k_axis = np.linspace(0, n_outer_iterations, n_outer_iterations+1)
    rect = matplotlib.patches.Rectangle((net.outer_iterations+0.5, 1.e-16), 7.5, 70, color='gray', alpha=0.2)
    ax.add_patch(rect)
    plt.semilogy(k_axis, avg_errors, label='R2N2 Test Error', linestyle='--', marker='d')
    plt.semilogy(k_axis, avg_gmres_, label='GMRES Error', linestyle='-', marker='^')
    plt.xticks(np.linspace(0, n_outer_iterations, n_outer_iterations+1))
    plt.xlabel(r'# Outer Iterations')
    plt.ylabel('Error')
    plt.xlim((-0.5, 10.5))
    plt.ylim((1.e-16, 70))
    plt.legend()
    plt.show()

def testAbRecNet():
    # Setup the network and load the weights
    net, _, _ = setupRecNet()
    weights = np.array([-0.13620847 , 0.08868813,  0.10030381 ,-1.53856325 , 0.76759586 , 2.0007581,
                        -1.13633718 ,-0.39309495 , 0.86202745, -0.03033491]) # Adam weights with random A and b
    
    # Load training data
    directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/'
    A_filename = 'A_Training_Data.npy'
    b_filename = 'b_Training_Data.npy'
    A_data = np.load(directory + A_filename)
    b_data = np.load(directory + b_filename)
    N_data = A_data.shape[2]

    # Run each rhs through the neural network
    n_outer_iterations = 10 # Does not need be the same as the number the network was trained on.
    n_inner_iterations = 4
    errors = np.zeros((N_data, n_outer_iterations+1))
    for n in range(N_data):
        A = A_data[:,:,n]
        rhs = b_data[:,n]
        samples = net.forward(weights, A, rhs, n_outer_iterations)

        for k in range(len(samples)):
            err = lg.norm(A.dot(samples[k]) - rhs)
            errors[n,k] = err

    # Solve the system Ax = rhs with gmres for all A and rhs
    gmres_errors = np.zeros((N_data, n_outer_iterations+1))
    for n in range(N_data):
        A = A_data[:,:,n]
        rhs = b_data[:,n]
        gmres_errors[n,0] = lg.norm(rhs)

        for k in range(1, n_outer_iterations+1):
            x, _ = slg.gmres(A, rhs, x0=np.zeros(rhs.size), maxiter=k, restart=n_inner_iterations, tol=0.0)
            gmres_errors[n,k] = lg.norm(A.dot(x) - rhs)
        print('gmres error', gmres_errors[n,n_outer_iterations], errors[n,n_outer_iterations])

    # Average the errors
    avg_errors = np.average(errors, axis=0)
    avg_gmres_ = np.average(gmres_errors, axis=0)

    # Plot the errors
    fig, ax = plt.subplots()  
    k_axis = np.linspace(0, n_outer_iterations, n_outer_iterations+1)
    rect = matplotlib.patches.Rectangle((net.outer_iterations+0.5, 1.e-16), 7.5, 70, color='gray', alpha=0.2)
    ax.add_patch(rect)
    plt.semilogy(k_axis, avg_errors, label='R2N2 Test Error', linestyle='--', marker='d')
    plt.semilogy(k_axis, avg_gmres_, label='GMRES Error', linestyle='-', marker='^')
    plt.xticks(np.linspace(0, n_outer_iterations, n_outer_iterations+1))
    plt.xlabel(r'# Outer Iterations')
    plt.ylabel('Error')
    plt.xlim((-0.5, 10.5))
    plt.ylim((1.e-16, 70))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    testAbRecNet()

