import sys
sys.path.append('../')

import torch as pt
import numpy.random as rd
import matplotlib.pyplot as plt

import ChemicalRoutines as cr

# Method parameters
M = 200
T = 100.0

# Initial Condition
seed = 100
rng = rd.RandomState(seed=seed)
eps = 0.01
U0 = 2.0
V0 = 0.75
U = U0 * pt.ones(M) + eps * pt.from_numpy(rng.normal(0.0, 1.0, M))
V = V0 * pt.ones(M) + eps * pt.from_numpy(rng.normal(0.0, 1.0, M))

# Run Lattice - Boltzmann
x = pt.cat((pt.unsqueeze(U, 0), pt.unsqueeze(V, 0)), dim=1)
phi = cr.LBM(x, T=T)
phi_U = phi[0,0:M]
phi_V = phi[0,M:]

# Plot found solution
x_array = pt.linspace(0.0, 1.0, M)
plt.plot(x_array.numpy(), phi_U.numpy(), label=r'$U(x)$', color='red')
plt.plot(x_array.numpy(), phi_V.numpy(), label=r'$V(x)$', color='blue')
plt.xlabel(r'$x$')
plt.ylabel(r'$U, V$')
plt.title('Lattice-Boltzmann Steady State')
plt.legend()
plt.show()
