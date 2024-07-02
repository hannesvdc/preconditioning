import torch as pt
import torch.optim as optim
import torch.optim.lr_scheduler as sch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from api.NewtonKrylovImpl import *

# Just some sanity pytorch settings
pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float64)

# The data impleementation and loader class
class ChemicalDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.rng = np.random.RandomState()
        
        self.N_data = 1024
        self.data_size = 40
        directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
        filename = 'Steady_State_LBM_dt=1e-4.npy'
        x0 = np.load(directory + filename).flatten()[0::10]
        self.data = pt.from_numpy(x0[None,:] + self.rng.normal(0.0, 1.0, size=(self.N_data, self.data_size)))

    def __len__(self):
        return self.N_data
	
    def __getitem__(self, idx):
        return self.data[idx,:], pt.zeros(self.data_size)
    
# Load the data in memory
print('Generating Training Data.')
batch_size = 64
dataset = ChemicalDataset()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Setup the PDE timestepper and psi flowmap function
# Parameters
d1 = 5.e-4; d2 = 0.06
dt = 1.e-4; T = 0.05; N = int(T / dt)
M = 20; dx = 1.0 / M

# Compute indices module M for periodic boundary conditions
def f_vectorized(x):
    U = x[:,0:M]; V = x[:, M:]
    ddU = (pt.roll(U, -1, dims=1) - 2.0*U + pt.roll(U, 1, dims=1)) / dx**2
    ddV = (pt.roll(V, -1, dims=1) - 2.0*V + pt.roll(V, 1, dims=1)) / dx**2
    f1 = d1*ddU + 1.0 - 2.0*U + U**2*V # f1 is a (N_data, M) array
    f2 = d2*ddV + 3.0         - U**2*V # f2 is a (N_data, M) array
    return pt.hstack((f1, f2))

# Apply right-hand side as update (with finite differences)
def PDE_Timestepper_vectorized(x):
	for _ in range(N):
		x = x + dt * f_vectorized(x) # the rhs is an (N_data, 2M) array
	return x
psi = lambda x: PDE_Timestepper_vectorized(x) - x # One-liner

# Initialize the Network and the Optimizer (Adam)
print('\nSetting Up the Newton-Krylov Neural Network.')
inner_iterations = 4
outer_iterations = 3
network = NewtonKrylovNetwork(psi, inner_iterations)
loss_fn = NewtonKrylovLoss(network, psi, outer_iterations)
adam_optimizer = optim.Adam(network.parameters(), lr=0.001)
scheduler = sch.StepLR(adam_optimizer, step_size=1000, gamma=0.1)
bfgs_optimizer = optim.LBFGS(network.parameters(), lr=0.01, max_iter=1000, line_search_fn=None)

# Apply whole dataset for debugging - To Do: remove later
network.forward(dataset.data)

# Training Routine
train_losses = []
train_counter = []
store_directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/NKNet/'
def train(epoch):
    network.train()
    for _, (data, _) in enumerate(train_loader):
        adam_optimizer.zero_grad()

        # Compute Loss. NewtonKrylovLoss takes care of network forwards
        loss = loss_fn(data)

        # Compute loss gradient and do one optimization step
        loss.backward()
        adam_optimizer.step()

    # Some housekeeping
    print('Train Epoch: {} \tLoss: {:.6f} \tLoss Gradient: {:.6f}'.format(epoch, loss.item(), pt.norm(network.inner_layer.weights.grad)))
    train_losses.append(loss.item())
    train_counter.append(epoch)
    pt.save(network.state_dict(), store_directory + 'model_chemical.pth')
    pt.save(adam_optimizer.state_dict(), store_directory + 'optimizer_chemical.pth')

# Do the actual training
print('\nStarting Adam Training Procedure...')
n_adam_epochs = 10000
try:
    for epoch in range(1, n_adam_epochs + 1):
        train(epoch)
        scheduler.step()
except KeyboardInterrupt:
    print('Aborting Adam Training. Starting fine-tuning with L-BFGS.')
passed_epochs = train_counter[-1]
pt.save(network.state_dict(), store_directory + 'model_adam_chemical.pth')
pt.save(adam_optimizer.state_dict(), store_directory + 'optimizer_adam_chemical.pth')

print('\nStarting BFGS Training Procedure...')
n_bfgs_epochs = 1000
x_lbfgs = pt.clone(dataset.data)
def bfgs_closure():
    bfgs_optimizer.zero_grad()
    loss = loss_fn(x_lbfgs)
    loss.backward()
    return loss
try:
    for epoch in range(1, n_bfgs_epochs + 1):
        print('BFGS Epoch', epoch)
        train_counter.append(passed_epochs + epoch)
        train_losses.append(loss_fn(x_lbfgs).item())
        bfgs_optimizer.step(bfgs_closure)

        pt.save(network.state_dict(), store_directory + 'model_chemical.pth')
        pt.save(bfgs_optimizer.state_dict(), store_directory + 'optimizer_chemical.pth')
except KeyboardInterrupt:
    print('Aborting Training. Plotting Training Convergence.')

# Show the training results
fig = plt.figure()
plt.semilogy(train_counter, train_losses, color='blue', label='Training Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()