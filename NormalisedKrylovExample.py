import torch as pt
import torch.optim as optim
import torch.optim.lr_scheduler as sch
import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import matplotlib.pyplot as plt

from api.NormalisedKrylovImpl import *

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
    
def getMatrix(id):
    if id == 1: # A well-conditioned symmetric matrix
        A_numpy = np.array([[1.392232, 0.152829, 0.088680, 0.185377, 0.156244],
                        [0.152829, 1.070883, 0.020994, 0.068940, 0.141251],
                        [0.088680, 0.020994, 0.910692,-0.222769, 0.060267],
                        [0.185377, 0.068940,-0.222769, 0.833275, 0.058072],
                        [0.156244, 0.141251, 0.060267, 0.058072, 0.735495]])
        A = pt.from_numpy(A_numpy)
    elif id == 2:
        seed = 100
        rng = rd.RandomState(seed=seed)
        U = rng.normal(0.0, 1.0, size=(5,5))
        V = rng.normal(0.0, 1.0, size=(5,5))
        Q_U, Q_V = lg.qr(U)[0], lg.qr(V)[0]
        A_numpy = np.matmul(Q_U, np.matmul(np.diag([1.0, 0.9, 0.8, 0.2, 0.1]), Q_V.T))
        A = pt.from_numpy(A_numpy)
        print(lg.cond(A_numpy))

    return A_numpy, A

def trainKrylovNN(id, A):

    print('Generating Training Data.')
    batch_size = 1024
    dataset = RHSDataset(M=5)
    train_loader = pt.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the Network and the Optimizer (Adam)
    print('\nSetting Up the Inverse Jacobian Neural Network.')
    inner_iterations = 4
    outer_iterations = 3
    network = NormalisedKrylovNetwork(A, inner_iterations)
    loss_fn = NormalisedKrylovLoss(network, outer_iterations)
    optimizer = optim.Adam(network.parameters(), lr=0.01)
    scheduler = sch.StepLR(optimizer, 20000, 0.1)

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
        print('Train Epoch: {} \tLoss: {:.6f} \tLoss Gradient: {:.6f}'.format(epoch, loss.item(), pt.norm(network.inner_layer.weights.grad)), '\tLearning Rate:', optimizer.param_groups[0]['lr'])
        pt.save(network.state_dict(), store_directory + 'normalised_krylov_nn_inner='+str(inner_iterations)+ '_id=' + str(id) + '.pth')
        train_losses.append(loss.item())
        train_counter.append(epoch)

    # Do the actual training
    print('\nStarting Adam Training Procedure...')
    n_epochs = 50000
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

def testKrylovNN(id, A_numpy, A):
    pt.set_grad_enabled(False)

    # Initialize the Network
    print('\nSetting Up the Krylov Neural Network.')
    inner_iterations = 4
    outer_iterations = 20
    store_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/NKNet/'
    filename = 'normalised_krylov_nn_inner='+str(inner_iterations)+'_id=' + str(id) +'.pth'
    network = NormalisedKrylovNetwork(A, inner_iterations)
    network.load_state_dict(pt.load(store_directory + filename))

    # Training Routine
    dataset = RHSDataset(M=A_numpy.shape[0])
    rhs_data = dataset.data
    x_data = pt.zeros_like(rhs_data)
    x0_data = np.copy(x_data.numpy())
    loss = lambda y: np.average(lg.norm(A_numpy.dot(y) - rhs_data.numpy().T, axis=0))
    errors = [loss(x_data.numpy().T)]
    gmres_errors = [loss(x0_data.T)]
    for outer_iter in range(outer_iterations):
        print(outer_iter)
        input = (x_data, rhs_data)
        x_data = network.forward(input)
        errors.append(loss(x_data.numpy().T))

        gmres_error = 0.0
        for n in range(rhs_data.shape[0]):
            gmres_sol, _ = slg.gmres(A_numpy, rhs_data.numpy()[n,:], x0=x0_data[n,:], restart=inner_iterations, maxiter=outer_iter+1, rtol=0.0)
            gmres_error += lg.norm(A_numpy.dot(gmres_sol) - rhs_data.numpy()[n,:])
        gmres_error /= x_data.shape[0]
        gmres_errors.append(gmres_error)
    print(errors)
    plt.semilogy(np.arange(0, len(errors)), np.array(errors), label='Krylov NN')
    plt.semilogy(np.arange(0, len(gmres_errors)), np.array(gmres_errors), label='GMRES')
    plt.xlabel(r'# Outer Iterations $k$')
    plt.ylabel(r'Backward Error')
    plt.legend()
    plt.title(r'$\frac{1}{N} \sum_{l=1}^N||A x_k - b_l ||$')
    plt.show()

if __name__ == '__main__':
    id = 1
    A_numpy, A = getMatrix(id)
    
    trainKrylovNN(id, A)
    testKrylovNN(id, A_numpy, A)