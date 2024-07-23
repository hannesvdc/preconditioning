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
        directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
        filename = 'Steady_State_LBM_dt=1e-4.npy'
        x0 = np.load(directory + filename).flatten()[::self.subsample]
        self.data = pt.from_numpy(x0[None,:] + self.rng.normal(0.0, self.scale, size=(self.N_data, self.data_size)))

    def __len__(self):
        return self.N_data
	
    def __getitem__(self, idx):
        return self.data[idx,:], pt.zeros(self.data_size)

class ChemicalLBMDataset(Dataset):
    def __init__(self, M):
        super().__init__()
        self.seed = 100
        self.scale = 0.1
        self.rng = np.random.RandomState(seed=self.seed)
        
        self.N_data = 1024
        self.M = M
        self.subsample = 200 // self.M
        self.data_size = 6 * self.M
        directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
        filename = 'Steady_State_LBM_dt=1e-4.npy'
        x0 = np.load(directory + filename).flatten()[::self.subsample]
        UV_data = pt.from_numpy(x0[None,:] + self.rng.normal(0.0, self.scale, size=(self.N_data, 2*self.M)))

        U = UV_data[:,0:M]
        V = UV_data[:,M:]
        self.data = pt.hstack((weights[0] * U, weights[1] * U, weights[2] * U, weights[0] * V, weights[1] * V, weights[2] * V))
        print(self.data.shape)

    def __len__(self):
        return self.N_data
	
    def __getitem__(self, idx):
        return self.data[idx,:], pt.zeros(self.data_size)

# Gillespie Model Parameters
# Parameters
d1 = 5.e-4
d2 = 0.06

# Time-Stepping Parameters
dt = 1.e-4

# LBM Method parameters
weights = np.array([1.0, 4.0, 1.0]) / 6.0 # D1Q3 weights
cs_quad = weights[0] + weights[2] #speed of sound squared
A = 1; B = 1; k1 = 1.0; k2 = 2.0; k3 = 3.0; k4 = 1.0 

# Compute indices module M for periodic boundary conditions
def f_pde_vectorized(x):
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
def PDE_Timestepper_vectorized(x, T):
    N = int(T / dt)
    for _ in range(N):
        x = x + dt * f_pde_vectorized(x) # the rhs is an (N_data, 2M) array
    return x
psi_pde = lambda x, T_psi: PDE_Timestepper_vectorized(x, T_psi) - x # One-liner

def Collisions_Reactions(f_1_U, f0_U, f1_U, f_1_V, f0_V, f1_V, relaxation_times):
    phi_U = f_1_U + f0_U + f1_U # Density of U (index 0, all space)
    phi_V = f_1_V + f0_V + f1_V # Density of V (index 0, all space)

    # Collisions, no additonal hermite polynommials needed because u(x, t) = 0 (BGK)
    c1_U = -relaxation_times[0] * (f1_U - weights[2] * phi_U)
    c1_V = -relaxation_times[1] * (f1_V - weights[2] * phi_V)
    c0_U = -relaxation_times[0] * (f0_U - weights[1] * phi_U)
    c0_V = -relaxation_times[1] * (f0_V - weights[1] * phi_V)
    c_1_U = -relaxation_times[0] * (f_1_U - weights[0] * phi_U)
    c_1_V = -relaxation_times[1] * (f_1_V - weights[0] * phi_V)

	# reaction term for activator (U)
    cross_UV = pt.multiply(pt.pow(phi_U, 2.0), phi_V)
    propensity_1 = k1*A - k2*phi_U + k4*cross_UV
    r1_U  = weights[2] * dt * propensity_1
    r0_U  = weights[1] * dt * propensity_1
    r_1_U = weights[0] * dt * propensity_1
	# reaction term for inibitor (V)
    propensity_2 = k3*B - k4*cross_UV
    r1_V  = weights[2] * dt * propensity_2
    r0_V  = weights[1] * dt * propensity_2
    r_1_V = weights[0] * dt * propensity_2

    return c_1_U, c0_U, c1_U, r_1_U, r0_U, r1_U, c_1_V, c0_V, c1_V, r_1_V, r0_V, r1_V

def _LatticeBM_Schnakenberg(f_1_U, f0_U, f1_U, f_1_V, f0_V, f1_V, relaxation_times):
	# Streaming in intermediate variable
    c_1_U, c0_U, c1_U, r_1_U, r0_U, r1_U, c_1_V, c0_V, c1_V, r_1_V, r0_V, r1_V \
                    = Collisions_Reactions(f_1_U, f0_U, f1_U, f_1_V, f0_V, f1_V, relaxation_times)
    f1star_U  = f1_U  + c1_U + r1_U
    f0star_U  = f0_U  + c0_U + r0_U
    f_1star_U = f_1_U + c_1_U + r_1_U
    f1star_V  = f1_V  + c1_V + r1_V
    f0star_V  = f0_V  + c0_V + r0_V
    f_1star_V = f_1_V + c_1_V + r_1_V
	
	# Updating, use periodc boundary conditions
    f0_U = pt.clone(f0star_U)
    f0_V = pt.clone(f0star_V)
    f1_U = pt.roll(f1star_U, 1, dims=1)
    f1_V = pt.roll(f1star_V, 1, dims=1)
    f_1_U = pt.roll(f_1star_U, -1, dims=1)
    f_1_V = pt.roll(f_1star_V, -1, dims=1)
   
    return f_1_U, f0_U, f1_U, f_1_V, f0_V, f1_V

# Implements D1Q3 Lattice-Boltzmann
# x is an (N_data, 6M) vector
def LBM(x, T):
	# Lattice Parameters
    M = x.shape[1] // 6
    dx = 1.0 / M
    N = int(T / dt)
    relaxation_times = pt.tensor([2.0/(1.0 + 2.0/cs_quad*d1*dt/dx**2), 2.0/(1.0 + 2.0/cs_quad*d2*dt/dx**2)])

	# Initial Condition for Lattice-Boltzmann.
    f_1_U = x[:, 0*M:1*M]
    f0_U  = x[:, 1*M:2*M]
    f1_U  = x[:, 2*M:3*M]
    f_1_V = x[:, 3*M:4*M]
    f0_V  = x[:, 4*M:5*M]
    f1_V  = x[:, 5*M:]

    # Do the actual time-stepping
    for _ in range(N):
        f_1_U, f0_U, f1_U, f_1_V, f0_V, f1_V = _LatticeBM_Schnakenberg(f_1_U, f0_U, f1_U,
                                                                       f_1_V, f0_V, f1_V,
                                                                       relaxation_times)

	# Concatenate all six concentrations horizontally
    return pt.hstack((f_1_U, f0_U, f1_U, f_1_V, f0_V, f1_V)) # equivalent of np.hstack
psi_lbm = lambda x, T_psi: LBM(x, T_psi) - x # One-liner

# Dt is the total step size (dt + extrapolation)
def EF_LBM(x, T, n, Dt):
    M = x.shape[1] // 2
    N = int(T / Dt)
    ext_factor = (Dt - n*dt) / dt # n*dt is the total microscopic integration time
    print(ext_factor)

    U, V = x[:,0:M], x[:,M:]
    for counter in range(N):
        print('T = ', counter*Dt)
        # Lift the current macroscopic state (x) to a new microscopic state
        f_1_U, f0_U, f1_U = weights[0] * U, weights[1] * U, weights[2] * U
        f_1_V, f0_V, f1_V = weights[0] * V, weights[1] * V, weights[2] * V
        y = pt.hstack((f_1_U, f0_U, f1_U, f_1_V, f0_V, f1_V))

		# Do n-1 microscopic steps, record the state, do one extra after that
        yp = LBM(y, (n-1)*dt)
        Up, Vp = yp[:,0:M] + yp[:,M:2*M] + yp[:,2*M:3*M], yp[:,3*M:4*M] + yp[:,4*M:5*M] + yp[:,5*M:]
        yq = LBM(yp, dt)
        Uq, Vq = yq[:,0:M] + yq[:,M:2*M] + yq[:,2*M:3*M], yq[:,3*M:4*M] + yq[:,4*M:5*M] + yq[:,5*M:]
        print(yp.isnan().any(), yq.isnan().any())

		# Extrapolation
        U = Uq + ext_factor * (Uq - Up)
        V = Vq + ext_factor * (Vq - Vp)

    x = pt.hstack((U, V))
    return x

psi_ef_lbm = lambda x, T_psi, dt, Dt: EF_LBM(x, T_psi, dt, Dt)