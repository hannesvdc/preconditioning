import torch as pt
import torch.optim as optim
import torch.optim.lr_scheduler as sch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from api.NewtonKrylovImpl import *
from ChemicalRoutines import psi

# Just some sanity pytorch settings
pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float64)

# The data impleementation and loader class
class ChemicalDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.seed = 100
        self.scale = 0.1
        self.rng = np.random.RandomState(seed=self.seed)
        
        self.N_data = 1024
        self.M = 25
        self.subsample = 200 // self.M
        self.data_size = 2 * self.M
        directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
        filename = 'Steady_State_LBM_dt=1e-4.npy'
        x0 = np.load(directory + filename).flatten()[0::self.subsample]
        self.data = pt.from_numpy(x0[None,:] + self.rng.normal(0.0, self.scale, size=(self.N_data, self.data_size)))

    def __len__(self):
        return self.N_data
	
    def __getitem__(self, idx):
        return self.data[idx,:], pt.zeros(self.data_size)
    
# Load the data in memory
print('Generating Training Data.')
batch_size = 64
dataset = ChemicalDataset()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the Network and the Optimizer (Adam)
print('\nSetting Up the Newton-Krylov Neural Network.')
inner_iterations = 4
outer_iterations = 3
network = NewtonKrylovNetwork(psi, inner_iterations)
loss_fn = NewtonKrylovLoss(network, psi, outer_iterations)
optimizer = optim.Adam(network.parameters(), lr=0.001)
scheduler = sch.StepLR(optimizer, step_size=1000, gamma=0.1)

# Apply whole dataset for debugging - To Do: remove later
network.forward(dataset.data)

# Training Routine
train_losses = []
train_counter = []
store_directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/NKNet/'
def train(epoch):
    network.train()
    for _, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()

        # Compute Loss. NewtonKrylovLoss takes care of network forwards
        loss = loss_fn(data)

        # Compute loss gradient and do one optimization step
        loss.backward()
        optimizer.step()

    # Some housekeeping
    print('Train Epoch: {} \tLoss: {:.6f} \tLoss Gradient: {:.6f}'.format(epoch, loss.item(), pt.norm(network.inner_layer.weights.grad)))
    train_losses.append(loss.item())
    train_counter.append(epoch)
    pt.save(network.state_dict(), store_directory + 'model_chemical.pth')
    pt.save(optimizer.state_dict(), store_directory + 'optimizer_chemical.pth')

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
pt.save(optimizer.state_dict(), store_directory + 'optimizer_adam_chemical.pth')

# Show the training results
fig = plt.figure()
plt.semilogy(train_counter, train_losses, color='blue', label='Training Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()