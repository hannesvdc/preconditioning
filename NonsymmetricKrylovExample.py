import torch as pt
import torch.optim as optim
import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import matplotlib.pyplot as plt

from api.KrylovImpl import *

pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float64)

class RHSDataset(pt.utils.data.Dataset):
    def __init__(self, M):
        super().__init__()

        self.seed = 100
        self.rng = np.random.RandomState(seed=self.seed)
        
        # Randomly generate the right-hand side dataset
        self.N_data = 1024
        self.data_size = M
        self.data = pt.from_numpy(self.rng.normal(0.0, 1.0, size=(self.N_data, self.data_size)))
        self.data.requires_grad = False

    def __len__(self):
        return self.N_data
	
    def __getitem__(self, idx):
        return self.data[idx, :]

def trainKrylovNN(matrix_type, cond_factor=10):
    M = 100
    rng = rd.RandomState()
    upper_s = rng.uniform(0.1, 0.2, 10) * cond_factor
    lower_s = rng.uniform(0.1, 0.2, 90)
    S = np.diag(np.concatenate((upper_s, lower_s)))
    U, _ = lg.qr((rng.uniform(size=(M, M)) - 5.) * 200)
    V, _ = lg.qr((rng.uniform(size=(M, M)) - 5.) * 200)
    if matrix_type == 'symmetric':
        A_numpy = U.dot(S).dot(U.T)
    else:
        A_numpy = U.dot(S).dot(V.T)
    print(lg.cond(A_numpy))
    A = pt.from_numpy(A_numpy).requires_grad_(False)

    print('Generating Training Data.')
    batch_size = 1024
    dataset = RHSDataset(M=A.shape[0])
    train_loader = pt.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize the Network and the Optimizer (Adam)
    print('\nSetting Up the Inverse Jacobian Neural Network.')
    inner_iterations = 15
    outer_iterations = 3
    network = KrylovNetwork(A, inner_iterations)
    loss_fn = KrylovLoss(network, outer_iterations)
    optimizer = optim.Adam(network.parameters(), lr=0.001, eps=1.e-2)

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
        pt.save(network.state_dict(), store_directory + 'krylovnn_' + matrix_type + '_cond=' + str(cond_factor) + '_inner='+str(inner_iterations)+'.pth')
        train_losses.append(loss.item())
        train_counter.append(epoch)

    # Do the actual training
    print('\nStarting Adam Training Procedure...')
    n_epochs = 50000
    try:
        for epoch in range(1, n_epochs+1):
            train(epoch)
            #scheduler.step()
    except KeyboardInterrupt:
        print('Terminating Training. Plotting Training Error Convergence.')

    # Show the training results
    plt.semilogy(train_counter, train_losses, color='blue', label='Training Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    return A_numpy

def testKrylovNN(A_numpy, matrix_type, cond_factor):
    pt.set_grad_enabled(False)

    # Initialize the Network
    print('\nSetting Up the Krylov Neural Network.')
    inner_iterations = 15
    outer_iterations = 10
    store_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/NKNet/'
    filename = 'krylovnn_' + matrix_type + '_cond=' + str(cond_factor) + '_inner='+str(inner_iterations)+'.pth'
    network = KrylovNetwork(pt.from_numpy(A_numpy), inner_iterations)
    network.load_state_dict(pt.load(store_directory + filename))

    # Training Routine
    dataset = RHSDataset(M=A_numpy.shape[0])
    rhs_data = dataset.data
    x_data = pt.zeros_like(rhs_data)
    loss = lambda y: np.sum(np.square(A_numpy.dot(y) - rhs_data.numpy().T)) / dataset.N_data
    errors = [loss(x_data.numpy().T)]
    for _ in range(outer_iterations):
        input = (x_data, rhs_data)
        x_data = network.forward(input)
        errors.append(loss(x_data.numpy().T))

    plt.semilogy(np.linspace(0, 11, 11), np.array(errors))
    plt.xlabel(r'# Outer Iterations $k$')
    plt.ylabel(r'Backward Error')
    plt.show()

if __name__ == '__main__':
    matrix_type = 'symmetric'
    cond_factor = 100
    A_numpy = trainKrylovNN(matrix_type, cond_factor)
    testKrylovNN(A_numpy, matrix_type, cond_factor)