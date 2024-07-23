import sys
sys.path.append('../')

import torch as pt
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

import ChemicalRoutines as cr

# Method parameters
M = 200
dt = 1.e-4 # microscopic LBM time step size
n = 5 # 5 microsteps during microscopic evolution
Dt = 2*n*dt
Tf = 100.0

# Sample Initial Condition from the Unstable Steady-State
seed = 100
rng = rd.RandomState(seed=seed)
eps = 0.00
ss_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
ss_filename = 'Steady_State_LBM_dt=1e-4.npy'
x_ss = np.load(ss_directory + ss_filename).flatten()
U_ss, V_ss = x_ss[0:M], x_ss[M:]
U = pt.from_numpy(U_ss + eps * rng.normal(0.0, 1.0, M))
V = pt.from_numpy(V_ss + eps * rng.normal(0.0, 1.0, M))

# Run Lattice - Boltzmann
x = pt.unsqueeze(pt.hstack((U, V)), dim=0)
print(x.shape)
x = cr.EF_LBM(x, T=Tf, n=n, Dt=Dt)
U, V = x[0,0:M], x[0,M:]
print(U)

# Plot found solution
x_array = pt.linspace(0.0, 1.0, M)
plt.plot(x_array.numpy(), U.numpy(), label=r'$U(x)$', color='red')
plt.plot(x_array.numpy(), V.numpy(), label=r'$V(x)$', color='blue')
#plt.plot(x_array.numpy(), U_ss, label=r'$U(x)$ Reference')
#plt.plot(x_array.numpy(), V_ss, label=r'$V(x)$ Reference')
plt.xlabel(r'$x$')
plt.ylabel(r'$U, V$')
plt.title('Lattice-Boltzmann Steady State')
plt.legend()
plt.show()
