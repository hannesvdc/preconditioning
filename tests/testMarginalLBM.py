import sys
sys.path.append('../')

import torch as pt
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

import ChemicalRoutines as cr

# Method parameters
M = 200
T_psi = 0.005
Tf = 100.0
N = int(Tf / T_psi)

# Sample Initial Condition from the Unstable Steady-State
seed = 100
rng = rd.RandomState(seed=seed)
eps = 0.01
U0 = 2.0
V0 = 0.75
U = U0 * pt.ones(M) + eps * pt.from_numpy(rng.normal(0.0, 1.0, M))
V = V0 * pt.ones(M) + eps * pt.from_numpy(rng.normal(0.0, 1.0, M))

# Transform the initial condition to the LBM format using weights
x = pt.unsqueeze(pt.hstack((U, V)), dim=0)
print(x.shape)

# Run Lattice - Boltzmann
for n in range(N):
    print('n =', n)
    x = cr._LBM_marginal(x, T=T_psi)
phi_U = x[0,0:M]
phi_V = x[0,M:]

# Plot found solution
# Load the exact steady state
ss_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
ss_filename = 'Steady_State_LBM_dt=1e-4.npy'
x_ss = np.load(ss_directory + ss_filename).flatten()
U_ss, V_ss = x_ss[0:M], x_ss[M:]
x_array = pt.linspace(0.0, 1.0, M)
plt.plot(x_array.numpy(), phi_U.numpy(), label=r'$U(x)$', color='red')
plt.plot(x_array.numpy(), phi_V.numpy(), label=r'$V(x)$', color='blue')
plt.plot(x_array.numpy(), U_ss, label=r'$U(x)$ Reference')
plt.plot(x_array.numpy(), V_ss, label=r'$V(x)$ Reference')
plt.xlabel(r'$x$')
plt.ylabel(r'$U, V$')
plt.title('Lattice-Boltzmann Steady State')
plt.legend()
plt.show()
