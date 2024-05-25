import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import scipy.optimize as opt
import scipy.optimize.nonlin as nl
import matplotlib as mpl
import matplotlib.pyplot as plt

import TrainNewtonKrylovNet as train

# Setup the network and load the weights
inner_iterations = 4
net, _, _ = train.setupRecNet(outer_iterations=3, inner_iterations=inner_iterations)
weights = np.array([ -5.65489262 , 15.67488836 ,  9.83137438 , 10.87824758 , 18.88908299,
                    -0.459666 ,  -12.37188497 ,-26.59916135 , -3.32415882 ,  3.3055943 ]) # Adam + BFGS refinement for 4 inner iterations
# weights = np.array([-1.03271415, -0.7997393,  -5.9540853,   0.61832089,  0.55027126, -1.09914459,
#                     -2.16602657,  4.75832453,  1.96480265, -0.7004273,  -0.73600679, -2.96956567,
#                     -2.41837951,  0.23372672, -1.85472313,  1.45412704, -0.02829014, -0.47087389,
#                      2.21695837,  2.03832011,  4.87277665,  1.15523701,  2.50459132, -0.36272781,
#                     -1.65029348, -2.6306855,  -2.83509759, -0.82244293, -0.6904838,  -1.69207527,
#                     -0.27236912, -4.96302391, -2.89757248,  1.03521323, -0.13689083,  2.47803962,
#                      0.58843259,  1.40852186, -1.57034324,  1.15022001, -1.58687977, -0.02086865,
#                      3.42095928,  2.00779577,  2.81886552,  0.05802382,  1.2304748,   0.74143769,
#                      0.39991601,  1.06206259,  0.68280461, -2.40568384,  2.13383043, -3.23630259,
#                      0.68182228]) # Adam + BFGS refinement for 10 inner iterations
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
avg_nk_errors = np.average(nk_errors, axis=0)

# Plot the errors
fig, ax = plt.subplots()  
k_axis = np.linspace(0, n_outer_iterations, n_outer_iterations+1)
rect = mpl.patches.Rectangle((net.outer_iterations+0.5, 1.e-8), 7.5, 70, color='gray', alpha=0.2)
ax.add_patch(rect)
plt.semilogy(k_axis, avg_errors, label='Neural Network Output Error', linestyle='--', marker='d')
plt.semilogy(k_axis, avg_nk_errors, label='Scipy Newton-Krylov Error', linestyle='--', marker='d')
plt.xticks(np.linspace(0, n_outer_iterations, n_outer_iterations+1))
plt.xlabel(r'# Outer Iterations $k$')
plt.ylabel('Error')
plt.xlim((-0.5,n_outer_iterations + 0.5))
plt.ylim((0.1*min(np.min(avg_nk_errors), np.min(avg_errors)),70))
plt.title(r'Newton-Krylov Neural Net')
plt.legend()
plt.show()