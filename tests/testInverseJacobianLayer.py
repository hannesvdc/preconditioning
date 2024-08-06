import sys
sys.path.append('../')

import torch as pt
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import ChemicalRoutines as cr
from api.PreconditionedNewtonKrylovImpl import *

pt.set_default_dtype(pt.float64)

class InverseJacobianNetwork(nn.Module):
    def __init__(self, inner_iterations):
        super(InverseJacobianNetwork, self).__init__()
        
        # setup the network psi function
        self.T_psi = 0.05
        self.F = lambda x: cr.psi_pde(x, self.T_psi)

        # This network is just one inner layer
        self.inner_layer = InverseJacobianLayer(self.F, inner_iterations)
        layer_list = [('layer_0', self.inner_layer)]
        self.layers = pt.nn.Sequential(OrderedDict(layer_list))

        # Check the number of parameters
        print('Number of Newton-Krylov Parameters:', sum(p.numel() for p in self.parameters()))

    # Input is a tuple containing the nonlinear iterate xk and the right-hand side rhs, 
    # both tensors of shape (N_data, N).
    def forward(self, input):
        return self.layers(input)
    
class InverseJacobianLoss(nn.Module):
    def __init__(self, network: InverseJacobianNetwork, outer_iterations, base_weight=4.0):
        super(InverseJacobianLoss, self).__init__()
        self.network = network
        self.outer_iterations = outer_iterations
        self.base_weight = base_weight

        self.F = self.network.inner_layer.F
        self.f = self.network.inner_layer.f

    # Input is a tuple containing the nonlinear iterate xk and the right-hand side rhs, 
    # both tensors of shape (N_data, N).
    def forward(self, data):
        # Load the data components
        xk  = data[0]
        rhs = data[1]
        w = pt.zeros_like(xk)
        F_value = self.F(xk)

        # Propagate the data and compute loss
        loss = 0.0
        for k in range(self.outer_iterations):
            loss_weight = self.base_weight ** k

            input = (xk, rhs, w)
            w = self.network.forward(input)
            loss += loss_weight * pt.sum(pt.square(self.f(w, rhs, xk, F_value))) # Sum over all data points (dim=0) and over all components (dim=1)

        # Average the loss and return
        N_data = rhs.shape[0]
        avg_loss = loss / N_data
        return avg_loss
    
class RHSDataset(Dataset):
    def __init__(self, M):
        super().__init__()

        self.seed = 100
        self.rng = np.random.RandomState(seed=self.seed)
        
        # Randomly generate the right-hand side dataset
        self.N_data = 1024
        self.M = M
        self.data_size = 2 * self.M
        self.rhs_data = pt.from_numpy(self.rng.normal(0.0, 1.0, size=(self.N_data, self.data_size)))
        self.rhs_data.requires_grad = False

        # Load the xk dataset
        self.xk_dataset = cr.ChemicalDataset(self.M)
        self.xk_data = self.xk_dataset.data
        self.xk_data.requires_grad = False

    def __len__(self):
        return self.N_data
	
    def __getitem__(self, idx):
        return (self.xk_data[idx, :], self.rhs_data[idx, :])
    
def trainInverseJacobianNetwork():
    pt.set_grad_enabled(True)

    # Load the data in memory
    print('Generating Training Data.')
    M = 200
    batch_size = 64
    dataset = RHSDataset(M)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the Network and the Optimizer (Adam)
    print('\nSetting Up the Newton-Krylov Neural Network.')
    inner_iterations = 25
    outer_iterations = 1
    network = InverseJacobianNetwork(inner_iterations)
    loss_fn = InverseJacobianLoss(network, outer_iterations)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    scheduler = sch.StepLR(optimizer, step_size=1000, gamma=0.1)

    # Training Routine
    train_losses = []
    train_counter = []
    store_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/NKNet/'
    def train(epoch):
        network.train()
        for _, data in enumerate(train_loader):
            optimizer.zero_grad()

            # Compute Loss. InverseJacobianLoss takes care of network forwards
            loss = loss_fn(data)

            # Compute loss gradient
            loss.backward()

            # Do one Adam optimization step
            optimizer.step()

        # Some housekeeping
        print('Train Epoch: {} \tLoss: {:.6f} \tLoss Gradient: {:.6f}'.format(epoch, loss.item(), pt.norm(network.inner_layer.weights.grad)))
        pt.save(network.state_dict(), store_directory + 'model_inverse_jacobian_inner='+str(inner_iterations)+'.pth')
        pt.save(optimizer.state_dict(), store_directory + 'optimizer_inverse_jacobian_inner='+str(inner_iterations)+'.pth')
        train_losses.append(loss.item())
        train_counter.append(epoch)

    # Do the actual training
    print('\nStarting Adam Training Procedure...')
    n_epochs = 10000
    try:
        for epoch in range(1, n_epochs+1):
            train(epoch)
            scheduler.step()
    except KeyboardInterrupt:
        print('Terminating Training. Plotting Training Error Convergence.')

    # Show the training results
    fig = plt.figure()
    plt.semilogy(train_counter, train_losses, color='blue', label='Training Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def testInverseJacobianNetwork():
    pt.set_grad_enabled(False)

    # Load the data in memory
    print('Generating Training Data.')
    M = 200
    dataset = RHSDataset(M)
    xk_data = dataset.xk_data
    rhs_data = dataset.rhs_data

    # Initialize the Network
    print('\nSetting Up the Newton-Krylov Neural Network.')
    inner_iterations = 10
    outer_iterations = 10
    network = InverseJacobianNetwork(inner_iterations)
    store_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/NKNet/'
    network.load_state_dict(pt.load(store_directory + 'model_inverse_jacobian_inner='+str(inner_iterations)+'.pth'))
    f = network.inner_layer.f
    F_value = network.inner_layer.F(xk_data)

    # Propagate the data through the network
    w = pt.zeros_like(rhs_data)
    averaged_errors = [pt.mean(pt.norm(f(w, rhs_data, xk_data, F_value), dim=1))]
    for k in range(outer_iterations):
        print('k =', k)

        input = (xk_data, rhs_data, w)
        w = network.forward(input)
        averaged_errors.append(pt.mean(pt.norm(f(w, rhs_data, xk_data, F_value), dim=1)))

    # Plot the test errors
    k_array = np.linspace(0.0, outer_iterations, outer_iterations+1)
    plt.semilogy(k_array, np.array(averaged_errors))
    plt.show()

if __name__ == '__main__':
    trainInverseJacobianNetwork()