import torch as pt
from torch.utils.data import Dataset

import numpy as np

# The data impleementation and loader class
class ChemicalDataset(Dataset):
    def __init__(self, M):
        super().__init__()
        self.seed = 100
        self.scale = 0.1
        self.rng = np.random.RandomState(seed=self.seed)
        
        self.N_data = 1024
        self.M = M
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

# Setup the PDE timestepper and psi flowmap function
# Parameters
d1 = 5.e-4
d2 = 0.06
dt = 1.e-4
T = 0.05
N = int(T / dt)

# Compute indices module M for periodic boundary conditions
def f_vectorized(x):
    M = x.shape[1] // 2
    dx = 1.0 / M

    U = x[:,0:M]
    V = x[:, M:]
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
psi_pde = lambda x: PDE_Timestepper_vectorized(x) - x # One-liner