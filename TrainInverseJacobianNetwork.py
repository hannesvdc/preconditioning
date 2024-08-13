import torch as pt
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as lg
import scipy.sparse.linalg as slg

import ChemicalRoutines as cr
from api.PreconditionedNewtonKrylovImpl import *

pt.set_default_dtype(pt.float64)

T_psi = 0.05
F = lambda x: cr.psi_pde(x, T_psi)
    
class InverseJacobianLoss(nn.Module):
    def __init__(self, layer: InverseJacobianLayer,
                       outer_iterations : int,
                       base_weight : float=4.0):
        super(InverseJacobianLoss, self).__init__()
        self.inner_layer = layer
        self.outer_iterations = outer_iterations
        self.baseweight = base_weight

        self.F = self.inner_layer.F
        self.f = self.inner_layer.f

        print('Number of trainable parameters:', sum(p.numel() for p in self.parameters()))

    def forward(self, data):
        xk  = data[0]
        rhs = data[1]
        w = pt.zeros_like(xk)
        Fxk = self.inner_layer.F(xk)

        loss = 0.0
        for k in range(self.outer_iterations):
            w = self.inner_layer.forward((xk, rhs, w))
            loss += self.baseweight**k * pt.sum(pt.square(self.f(xk, w, rhs, Fxk)))

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
        self.xk_dataset = cr.ChemicalDataset(self.M, scale=0.0)
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
    M = 50
    batch_size = 64
    dataset = RHSDataset(M)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the Network and the Optimizer (Adam)
    print('\nSetting Up the Inverse Jacobian Neural Network.')
    inner_iterations = 4
    outer_iterations = 8
    network = InverseJacobianLayer(F, inner_iterations)
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
        print('Train Epoch: {} \tLoss: {:.6f} \tLoss Gradient: {:.6f}'.format(epoch, loss.item(), pt.norm(network.weights.grad)))
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
    plt.semilogy(train_counter, train_losses, color='blue', label='Training Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def testGMRESConvergence():
    pt.set_grad_enabled(False)

    # Load the data
    M = 50
    dataset = RHSDataset(M)
    xk = dataset.xk_data[0,:].numpy()
    xk_pt = pt.unsqueeze(dataset.xk_data[0,:], dim=0)
    rhs = dataset.rhs_data.numpy()

    # Setup the Jacobian-vector products
    eps = 1.e-8
    G = lambda x: F(pt.from_numpy(x[np.newaxis,:]))[0,:].numpy()
    G_value = G(xk)
    def jvp(v):
        norm_v = lg.norm(v)
        normed_v = v / norm_v
        return norm_v * (G(xk + eps * normed_v) - G_value) / eps
    A = slg.LinearOperator(shape=(2*M,2*M,), matvec=jvp)

    # Load inversejacobiannetwork with optimized parameters
    inner_iterations = 95
    network = InverseJacobianLayer(F, inner_iterations)
    store_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/NKNet/'
    network.load_state_dict(pt.load(store_directory + 'model_inverse_jacobian_inner='+str(inner_iterations)+'.pth'))
    network.computeFValue(pt.unsqueeze(dataset.xk_data[0,:], dim=0))

    # Run gmres with 95 Krylov vectors for each rhs
    x0 = eps * np.ones_like(xk)
    errors = []
    for index in range(rhs.shape[0]):
        _rhs = rhs[index,:]
        x, _ = slg.gmres(A, _rhs, x0=np.copy(x0), restart=95, maxiter=1)
        backward_error = np.sum(np.square(A.matvec(x) - _rhs))
        errors.append(backward_error)

        y = network.forward((xk_pt, pt.from_numpy(_rhs[np.newaxis,:])))
        backward_error_y = np.sum(np.square(A.matvec(y[0].numpy()) - _rhs))
        print('index', index, backward_error, backward_error_y)
    errors = np.array(errors)

    print('Average Squared Residual', np.average(errors))

def testInverseJacobianNetwork():
    pt.set_grad_enabled(False)

    # Load the data in memory
    print('Generating Training Data.')
    M = 50
    dataset = RHSDataset(M)
    xk_data = dataset.xk_data
    rhs_data = dataset.rhs_data

    # Initialize the Network
    print('\nSetting Up the Newton-Krylov Neural Network.')
    inner_iterations = 4
    outer_iterations = 10
    network = InverseJacobianLayer(F, inner_iterations)
    store_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/NKNet/'
    network.load_state_dict(pt.load(store_directory + 'model_inverse_jacobian_inner='+str(inner_iterations)+'.pth'))
    Fxk_data = network.F(xk_data)
    f = network.f

    # Propagate the data through the network
    w = pt.zeros_like(xk_data)
    averaged_errors = [pt.mean(pt.norm(f(xk_data, w, rhs_data, Fxk_data), dim=1))]
    for k in range(outer_iterations):
        print('k =', k)

        input = (xk_data, rhs_data, w)
        w = network.forward(input)
        averaged_errors.append(pt.mean(pt.norm(f(xk_data, w, rhs_data, Fxk_data), dim=1)))

    # Plot the test errors
    k_array = np.linspace(0.0, outer_iterations, outer_iterations+1)
    plt.semilogy(k_array, np.array(averaged_errors))
    plt.show()

if __name__ == '__main__':
    testInverseJacobianNetwork()