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
U = pt.unsqueeze(U0 * pt.ones(M) + eps * pt.from_numpy(rng.normal(0.0, 1.0, M)), dim=0)
V = pt.unsqueeze(V0 * pt.ones(M) + eps * pt.from_numpy(rng.normal(0.0, 1.0, M)), dim=0)

# Run Lattice - Boltzmann
weights = np.array([1.0, 4.0, 1.0]) / 6.0 # D1Q3 weights
for n in range(N):
    print('n =', n)
    f_1_U, f0_U, f1_U = weights[0] * U, weights[1] * U, weights[2] * U
    f_1_V, f0_V, f1_V = weights[0] * V, weights[1] * V, weights[2] * V
    x = pt.hstack((f_1_U, f0_U, f1_U, f_1_V, f0_V, f1_V))

    x = cr.LBM(x, T=T_psi)
    
    U = x[:,0:M]     + x[:,M:2*M]   + x[:,2*M:3*M]
    V = x[:,3*M:4*M] + x[:,4*M:5*M] + x[:,5*M:]

# Plot found solution
# Load the exact steady state
ss_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
ss_filename = 'Steady_State_LBM_dt=1e-4.npy'
x_ss = np.load(ss_directory + ss_filename).flatten()
U_ss, V_ss = x_ss[0:M], x_ss[M:]
x_array = pt.linspace(0.0, 1.0, M)
plt.plot(x_array.numpy(), U.numpy()[0,:], label=r'$U(x)$', color='red')
plt.plot(x_array.numpy(), V.numpy()[0,:], label=r'$V(x)$', color='blue')
plt.plot(x_array.numpy(), U_ss, label=r'$U(x)$ Reference')
plt.plot(x_array.numpy(), V_ss, label=r'$V(x)$ Reference')
plt.xlabel(r'$x$')
plt.ylabel(r'$U, V$')
plt.title('Lattice-Boltzmann Steady State')
plt.legend()
plt.show()
