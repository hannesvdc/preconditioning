import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import matplotlib as mpl
import matplotlib.pyplot as plt

import TrainNewtonKrylovNet as train

# Setup the network and load the weights
net, _, _ = train.setupRecNet()
weights = np.array([ -5.65489262 , 15.67488836 ,  9.83137438 , 10.87824758 , 18.88908299,
                    -0.459666 ,  -12.37188497 ,-26.59916135 , -3.32415882 ,  3.3055943 ]) # Adam + BFGS refinement
    
# Generate test data. Same distribution as training data. Test actual training data next
m = 10
N_data = 1000
rng = rd.RandomState()
x0_data = rng.normal(1.0, np.sqrt(0.2), size=(m, N_data))

# Run each rhs through the neural network
n_outer_iterations = 10 # Does not need be the same as the number the network was trained on.
n_inner_iterations = 4
errors = np.zeros((N_data, n_outer_iterations+1))
for n in range(N_data):
    samples = net.forward(x0_data[:,n], weights, n_outer_iterations)

    for k in range(len(samples)):
        err = lg.norm(net.f(samples[k]))
        errors[n,k] = err

# Average the errors
avg_errors = np.average(errors, axis=0)

# Plot the errors
fig, ax = plt.subplots()  
k_axis = np.linspace(0, n_outer_iterations, n_outer_iterations+1)
rect = mpl.patches.Rectangle((net.outer_iterations+0.5, 1.e-16), 7.5, 70, color='gray', alpha=0.2)
ax.add_patch(rect)
plt.semilogy(k_axis, avg_errors, label='R2N2 Test Error', linestyle='--', marker='d')
plt.xticks(np.linspace(0, n_outer_iterations, n_outer_iterations+1))
plt.xlabel(r'# Outer Iterations')
plt.ylabel('Error')
plt.title(r'Newton-Krylov Neural Net')
plt.legend()
plt.show()