import sys
sys.path.append('../')
import tracemalloc

import torch as pt
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

import ChemicalRoutines as cr

# Method parameters
M = 200
dt = 1.e-4 # microscopic LBM time step size
n_micro = 1000
dT_min = 0.0
dT_max = 1.e-1
Tf = 100.0
tolerance = 1.e-3

# Sample Initial Condition from the Unstable Steady-State
seed = 100
rng = rd.RandomState(seed=seed)
eps = 0.1
ss_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
ss_filename = 'Steady_State_LBM_dt=1e-4.npy'
x_ss = np.load(ss_directory + ss_filename).flatten()
U_ss, V_ss = x_ss[0:M], x_ss[M:]
x_array = np.linspace(0.0, 1.0, M)
N_data = 2
U = pt.from_numpy(U_ss + eps * rng.normal(0.0, 1.0, size=(N_data, M)))
V = pt.from_numpy(V_ss + eps * rng.normal(0.0, 1.0, size=(N_data, M)))

# Run original Lattice-Boltzmann Method
#print('Running LBM')
#weights = pt.tensor([1.0, 4.0, 1.0]) / 6.0
#f_1_U, f0_U, f1_U = weights[0] * U, weights[1] * U, weights[2] * U
#f_1_V, f0_V, f1_V = weights[0] * V, weights[1] * V, weights[2] * V
#y = pt.unsqueeze(pt.hstack((f_1_U, f0_U, f1_U, f_1_V, f0_V, f1_V)), dim=0)
#y = cr.LBM(y, Tf)
#U_LBM, V_LBM = y[0, 0:M] + y[0, M:2*M] + y[0, 2*M:3*M], y[0, 3*M:4*M] + y[0, 4*M:5*M] + y[0, 5*M:]

# Run Lattice - Boltzmann
print('Running EqF-LBM')
tracemalloc.start()
#x = pt.unsqueeze(pt.hstack((U, V)), dim=0)
x = pt.hstack((U, V))
#x = cr.equation_free_LBM(x, Tf, n_micro, dT_min, dT_max, tolerance)
y = cr.psi_ef_lbm(x, 0.5, n_micro, dT_min, dT_max, tolerance)
first_size, first_peak = tracemalloc.get_traced_memory()
print(f"{first_size=}, {first_peak=}")
tracemalloc.stop()
U, V = y[0,0:M], y[0,M:]
#print('Psi EqF-LBM', pt.norm(cr.psi_ef_lbm(x, 1.0, n_micro, dT_min, dT_max, tolerance))) # Gets a value of 0.0052, which is nice!

# Plot found solution
plt.plot(x_array, U.numpy(), label=r'$U(x) \ EqF-LBM$')
plt.plot(x_array, V.numpy(), label=r'$V(x) \ EqF-LBM$')
#plt.plot(x_array, U_LBM.numpy(), label=r'$U(x) \ LBM$')
#plt.plot(x_array, V_LBM.numpy(), label=r'$V(x) \ LBM$')
plt.xlabel(r'$x$')
plt.ylabel(r'$U, V$')
plt.title('Lattice-Boltzmann Steady State')
plt.legend()
plt.show()
