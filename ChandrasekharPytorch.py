import torch as pt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt
import scipy.optimize.nonlin as nl
import matplotlib as mpl
import matplotlib.pyplot as plt

from api.NewtonKrylovImpl import *

# Just some sanity pytorch settings
pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float64)

# The data impleementation and loader class
class ChandrasekharDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.rng = np.random.RandomState()
        
        self.m = 10
        self.N_data = 1024
        self.data = pt.from_numpy(self.rng.normal(1.0, np.sqrt(0.2), size=(self.N_data, self.m)))

    def __len__(self):
        return self.N_data
	
    def __getitem__(self, idx):
        return self.data[idx,:], pt.zeros(self.m)
    
# Just some sanity pytorch settings
pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float64)

# Load the data in memory
print('Generating Training Data.')
batch_size = 64
dataset = ChandrasekharDataset()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Setup the H-function
c = 0.875
m = 10
mu = (pt.arange(1, m+1, 1) - 0.5) / m
def computeAc():
    Ac = pt.zeros((10,10))
    for i in range(10):
        for j in range(10):
            Ac[i,j] = mu[i] / (mu[i] + mu[j])
    return 0.5 * c/m * Ac
Ac_numpy = computeAc().numpy()
Ac = computeAc().transpose(0, 1) # Transpose because pytorch stores data in rows
H = lambda x: x + 1.0 / (1.0 + pt.mm(x, Ac))
H_vector = lambda x: x + 1.0 / (1.0 + np.dot(Ac_numpy, x))

# Initialize the Network and the Optimizer (Adam)
print('\nSetting Up the Newton-Krylov Neural Network.')
inner_iterations = 4
outer_iterations = 3
network = NewtonKrylovNetwork(H, inner_iterations)
loss_fn = NewtonKrylovLoss(network, H, outer_iterations)
optimizer = optim.Adam(network.parameters())
network.forward(dataset.data)

# Training Routine
train_losses = []
train_counter = []
log_rate = 4
store_directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/NKNet/'
def train(epoch):
    network.train()
    for _, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()

        # Compute Loss, NKLoss takes care of network forwards
        loss = loss_fn(data)

        # Compute loss gradient and do one optimization step
        loss.backward()
        optimizer.step()

    # Some housekeeping
    print('Train Epoch: {} \tLoss: {:.6f} \tLoss Gradient: {:.6f}'.format(epoch, loss.item(), pt.norm(network.inner_layer.weights.grad)))
    train_losses.append(loss.item())
    train_counter.append(epoch)
    pt.save(network.state_dict(), store_directory + 'model_chandrasekhar.pth')
    pt.save(optimizer.state_dict(), store_directory + 'optimizer_chandrasekhar.pth')

# Do the actual training
print('\nStarting Training Procedure...')
n_epochs = 200000
try:
    for epoch in range(1, n_epochs + 1):
        train(epoch)
except KeyboardInterrupt:
    print('Aborting Training. Plotting Training Convergence.')

# Show the training results
fig = plt.figure()
plt.semilogy(train_counter, train_losses, color='blue', label='Training Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Testing Routine
test_dataset = ChandrasekharDataset()
N_test_data = len(test_dataset)
x = pt.clone(test_dataset.data)

# Run each rhs through the neural network
n_outer_iterations = 10
nn_errors = pt.zeros((N_test_data, n_outer_iterations+1))
nk_errors = pt.zeros((N_test_data, n_outer_iterations+1))
nn_errors[:,0] = pt.norm(H(x), dim=1)
for k in range(1, n_outer_iterations+1):
    x = network.forward(x)
    nn_errors[:,k] = pt.norm(H(x), dim=1)
for n in range(N_test_data):
    x0 = test_dataset.data[n,:].numpy()
    for k in range(n_outer_iterations+1):
        try:
            x_out = opt.newton_krylov(H_vector, x0, rdiff=1.e-8, iter=k, maxiter=k, method='gmres', inner_maxiter=1, outer_k=0, line_search=None)
        except nl.NoConvergence as e:
            x_out = e.args[0]
        nk_errors[n,k] = lg.norm(H_vector(x_out))

# Average the errors
avg_nn_errors = pt.mean(nn_errors, dim=0)
avg_nk_errors = pt.mean(nk_errors, dim=0)

# Plot the errors
with pt.no_grad():
    fig, ax = plt.subplots()  
    k_axis = pt.linspace(0, n_outer_iterations, n_outer_iterations+1)
    rect = mpl.patches.Rectangle((loss_fn.outer_iterations+0.5, 1.e-8), 7.5, 70, color='gray', alpha=0.2)
    ax.add_patch(rect)
    plt.semilogy(k_axis, avg_nn_errors, label=r'Newton-Krylov Network with $4$ Inner Iterations', linestyle='--', marker='d')
    plt.semilogy(k_axis, avg_nk_errors, label=r'Scipy with $4$ Krylov Vectors', linestyle='--', marker='d')
    plt.xticks(pt.linspace(0, n_outer_iterations, n_outer_iterations+1))
    plt.xlabel(r'# Outer Iterations $k$')
    plt.ylabel(r'$|H(x_k)|$')
    plt.xlim((-0.5,n_outer_iterations + 0.5))
    plt.ylim((0.1*min(pt.min(avg_nk_errors), pt.min(avg_nn_errors)),70))
    plt.title(r'Function Value $|H(x_k)|$')
    plt.legend()
    plt.show()