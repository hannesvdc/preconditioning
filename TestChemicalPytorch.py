import torch as pt

import numpy as np
import matplotlib.pyplot as plt

from api.NewtonKrylovImpl import *
from ChemicalRoutines import psi

# Just some sanity pytorch settings
pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float64)

# Load the stored steady state and perturb it
M = 20
subsample = 200 // M
directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
filename = 'Steady_State_LBM_dt=1e-4.npy'
x0 = np.load(directory + filename).flatten()[0::subsample]

N_data = 1
scale = 0.1
rng = np.random.RandomState()
data = pt.from_numpy(x0[None,:] + rng.normal(0.0, scale, size=(N_data, 2*M)))

# Load the network state
store_directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/NKNet/'
inner_iterations = 4
outer_iterations = 10
network = NewtonKrylovNetwork(psi, inner_iterations)
network.load_state_dict(pt.load(store_directory + 'model_adam_chemical_M=' + str(M) + '.pth'))

# Propagate the data 10 times
x = pt.clone(data)
for n in range(outer_iterations):
    x = network.forward(x)
print(pt.norm(psi(x)))

# Show the found steady-state (hopefully)
U = x[0,0:20]
V = x[0,20:]
x_array = np.linspace(0.0, 1.0, 20)
plt.plot(x_array, U.detach().numpy(), label=r'$U(x)$', color='red')
plt.plot(x_array, V.detach().numpy(), label=r'$V(x)$', color='blue')
plt.xlabel(r'$x$')
plt.title(r'Steady-State PDE')
plt.legend()
plt.show()
