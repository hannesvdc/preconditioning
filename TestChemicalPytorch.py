import torch as pt

import numpy as np
import matplotlib.pyplot as plt

from api.NewtonKrylovImpl import *
from ChemicalRoutines import psi, ChemicalDataset

# Just some sanity pytorch settings
pt.set_grad_enabled(False)
pt.set_default_dtype(pt.float64)

# Load the stored steady state and perturb it
M = 25
dataset = ChemicalDataset(M=M)
x = pt.clone(dataset.data)

# Load the network state
store_directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/NKNet/'
inner_iterations = 4
outer_iterations = 10
network = NewtonKrylovNetwork(psi, inner_iterations)
network.load_state_dict(pt.load(store_directory + 'model_chemical.pth'))

# Propagate the data 10 times
errors = pt.zeros(x.shape[0], outer_iterations+1)
errors[:,0] = pt.norm(psi(x), dim=1)
for n in range(outer_iterations):
    x = network.forward(x)
    errors[:,n+1] = pt.norm(psi(x), dim=1)

# Find the index with minimal error
index = pt.argmin(errors[:, outer_iterations])
U = x[index, 0:M]
V = x[index, M:]
x_array = np.linspace(0.0, 1.0, M)
plt.plot(x_array, U.detach().numpy(), label=r'$U(x)$', color='red')
plt.plot(x_array, V.detach().numpy(), label=r'$V(x)$', color='blue')
plt.xlabel(r'$x$')
plt.title(r'Steady-State PDE')
plt.legend()

# Average the erros and show convergence
mse = pt.mean(errors, dim=0)
outers = pt.arange(0, outer_iterations+1)
plt.figure()
plt.semilogy(outers, mse, label=r'MSE')
plt.legend()
plt.show()
