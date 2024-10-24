import torch as pt
import torch.optim as optim
import torch.optim.lr_scheduler as sch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from api.PreconditionedNewtonKrylovImpl import *
from ChemicalRoutines import psi_eqfree_tensor, psi_pde, ChemicalDataset

# Just some sanity pytorch settings
pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float64)

# Setup the equation-free function
n_micro = 1000
dT = 0.1
T_psi = 0.5
psi_ef = lambda x: psi_eqfree_tensor(x, T_psi, n_micro, dT)

# Setup the macroscopic preconditioning function
T_pde = 0.05
psi_macro = lambda x: psi_pde(x, T_pde)
def M_generator(xk, F_value):
    N_data = xk.shape[0]
    ds = xk.shape[1]

    I = pt.eye(ds)
    M = pt.zeros((N_data, ds, ds))

    eps = 1.e-8
    for i in range(ds):
        e = (I[:, i])[None,:] # Shape (1, ds) but broadcastable
        F_diff = (psi_macro(xk + eps * e) - F_value) / eps
        M[:,:,i] = F_diff

    return M


# Load the data in memory
print('Generating Training Data.')
M = 200
batch_size = 2
dataset = ChemicalDataset(M=M, scale=0.1)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the Preconditioned NK Network and the Optimizer (Adam)
print('\nSetting Up the Newton-Krylov Neural Network.')
inner_iterations = 4
outer_iterations = 3
network = PreconditionedNewtonKrylovNetwork(psi_ef, inner_iterations, M_generator)
loss_fn = PreconditionedNewtonKrylovLoss(network, outer_iterations)
optimizer = optim.Adam(network.parameters(), lr=1.e-4)
scheduler = sch.StepLR(optimizer, step_size=1000, gamma=0.1)

# Training Routine
train_losses = []
train_counter = []
store_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/NKNet/'
def train(epoch):
    network.train()
    for batch_id, (data, _) in enumerate(train_loader):
        print('batch', batch_id)
        optimizer.zero_grad()

        # Compute Loss. NewtonKrylovLoss takes care of network forwards
        loss = loss_fn(data)
        if pt.any(pt.isnan(loss)):
            print('NaN-values detected during training.')
            exit(1)

        # Compute loss gradient
        loss.backward()

        # Do one Adam optimization step
        optimizer.step()
        print(loss.item(), pt.norm(network.inner_layer.weights.grad))

    # Some housekeeping
    print('Train Epoch: {} \tLoss: {:.6f} \tLoss Gradient: {:.6f}'.format(epoch, loss.item(), pt.norm(network.inner_layer.weights.grad)))
    train_losses.append(loss.item())
    train_counter.append(epoch)
    pt.save(network.state_dict(), store_directory + 'model_preconditioned_chemical_M='+str(M)+'_inner='+str(inner_iterations)+'.pth')
    pt.save(optimizer.state_dict(), store_directory + 'optimizer_preconditioned_chemical_M='+str(M)+'_inner='+str(inner_iterations)+'.pth')

# Do the actual training
print('\nStarting Adam Training Procedure...')
n_epochs = 10000
try:
    for epoch in range(1, n_epochs+1):
        train(epoch)
        #scheduler.step()
except KeyboardInterrupt:
    print('Terminating Training. Plotting Training Error Convergence.')

# Show the training results
fig = plt.figure()
plt.semilogy(train_counter, train_losses, color='blue', label='Training Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()