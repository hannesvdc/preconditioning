import sys
sys.path.append('../')

import torch as pt
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

import ChemicalRoutines as cr

pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float64)

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
N_data = 64
U = pt.from_numpy(U_ss + eps * rng.normal(0.0, 1.0, size=(N_data, M)))
V = pt.from_numpy(V_ss + eps * rng.normal(0.0, 1.0, size=(N_data, M)))

# Run Equation-Free Lattice-Boltzmann
x = pt.hstack((U, V))
x = cr.equation_free_LBM_tensor(x, Tf, n_micro, dT_max)
U, V = x[0,0:M], x[0,M:]
print('Psi EqF-LBM', pt.norm(cr.psi_eqfree_tensor(x, 0.5, n_micro, dT_max)) / N_data)

# Plot found solution
plt.plot(x_array, U.numpy(), label=r'$U(x)$ Equation-Free')
plt.plot(x_array, V.numpy(), label=r'$V(x)$ Equation-Free')
plt.plot(x_array, U_ss, label=r'$U(x)$ Steady-State')
plt.plot(x_array, V_ss, label=r'$V(x)$ Steady-State')
plt.xlabel(r'$x$')
plt.ylabel(r'$U, V$')
plt.title('Lattice-Boltzmann Steady State')
plt.legend()
plt.show()
