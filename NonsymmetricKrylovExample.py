import torch as pt
import torch.optim as optim
import torch.optim.lr_scheduler as sch
import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import matplotlib.pyplot as plt

from api.KrylovImpl import *

pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float64)

M = 100
cond_A = 1.0
rng = rd.RandomState()
upper_s = rng.uniform(1.0, 2.0, 10)
lower_s = rng.uniform(0.1, 0.2, 90)
S = np.diag(np.concatenate((upper_s, lower_s)))
U, _ = lg.qr((np.random.rand(M, M) - 5.) * 200)
A = U.dot(S).dot(U.T)
A = pt.from_numpy(A).requires_grad_(False)

class RHSDataset(pt.utils.data.Dataset):
    def __init__(self):
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

print('Generating Training Data.')
batch_size = 1024
dataset = RHSDataset()
train_loader = pt.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Initialize the Network and the Optimizer (Adam)
print('\nSetting Up the Inverse Jacobian Neural Network.')
inner_iterations = 15
outer_iterations = 3
network = KrylovNetwork(A, inner_iterations)
loss_fn = KrylovLoss(network, outer_iterations)
optimizer = optim.Adam(network.parameters(), lr=0.001)
scheduler = sch.StepLR(optimizer, step_size=5000, gamma=0.1)

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
    pt.save(network.state_dict(), store_directory + 'krylovnn_symmetric_cond=10_inner='+str(inner_iterations)+'.pth')
    train_losses.append(loss.item())
    train_counter.append(epoch)

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
plt.semilogy(train_counter, train_losses, color='blue', label='Training Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()