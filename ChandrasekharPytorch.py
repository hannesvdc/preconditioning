import torch as pt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
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
        return self.input_data[idx,:], pt.zeros(self.m)
    
# Just some sanity pytorch settings
pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float64)

# Load the data in memory
print('Generating Training Data.')
batch_size = 256
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
Ac = computeAc()
H = lambda x: x + 1.0 / (1.0 + pt.dot(Ac, x.transpose()).transpose())

# Initialize the Network and the Optimizer (Adam)
print('\nSetting Up the Newton-Krylov Neural Network.')
inner_iterations = 4
outer_iterations = 3
network = NewtonKrylovNetwork(H, inner_iterations)
loss_fn = NewtonKrylovLoss(network, H, outer_iterations)
optimizer = optim.Adam(network.parameters())
network.forward(dataset.input_data)

# Training Routine
train_losses = []
train_counter = []
log_rate = 100
def train(epoch):
    network.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()

        # Compute Loss
        output = network(data)
        loss = loss_fn(output)

        # Compute loss gradient and do one optimization step
        loss.backward()
        optimizer.step()

        # Some housekeeping
        if batch_idx % log_rate == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

# Do the actual training
print('\nStarting Training Procedure...')
n_epochs = 50000
try:
    for epoch in range(1, n_epochs + 1):
        train(epoch)
except KeyboardInterrupt:
    print('Aborting Training. Plotting Training Convergence.')

# Show the training results
fig = plt.figure()
plt.semilogy(train_counter, train_losses, color='blue', label='Training Loss')
plt.legend()
plt.xlabel('Number of training examples seen')
plt.ylabel('Loss')
plt.show()