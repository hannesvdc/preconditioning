import torch as pt

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from api.NewtonKrylovImpl import *
from ChemicalRoutines import psi_eqfree_tensor, ChemicalDataset

# Just some sanity pytorch settings
pt.set_grad_enabled(False)
pt.set_default_dtype(pt.float64)

# Global variables to be shared by all testing routines
# Create function to solve
n_micro = 1000
dT = 0.1
T_psi = 0.5
psi = lambda x: psi_eqfree_tensor(x, T_psi, n_micro, dT)

# Load the stored steady state and perturb it
M = 200
dataset = ChemicalDataset(M=M)
x = pt.clone(dataset.data)

# Load the network state
store_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/R2N2/NKNet/'
inner_iterations = 4
outer_iterations = 10
network = NewtonKrylovNetwork(psi, inner_iterations)
network.load_state_dict(pt.load(store_directory + 'model_eqflbm_chemical_M='+str(M)+'_inner='+str(inner_iterations)+'.pth'))

# Load the exact steady state
ss_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
ss_filename = 'Steady_State_LBM_dt=1e-4.npy'
x_ss = np.load(ss_directory + ss_filename).flatten()[::dataset.subsample]
U_ss, V_ss = x_ss[0:M], x_ss[M:]

# Propagate the data 10 times
errors = pt.zeros(x.shape[0], outer_iterations+1)
errors[:,0] = pt.norm(psi(x), dim=1)
for n in range(outer_iterations):
    x = network.forward(x)
    errors[:,n+1] = pt.norm(psi(x), dim=1)

# Find the index with minimal error
index = pt.argmin(errors[:, outer_iterations])
U, V = x[index, 0:M], x[index,M:]
x_array = np.linspace(0.0, 1.0, M)
plt.plot(x_array, U_ss, label=r'Steady State $U(x)$', color='green')
plt.plot(x_array, V_ss, label=r'Steady State $V(x)$', color='orange')
plt.plot(x_array, U.detach().numpy(), label=r'Newton-Krylov NN $U(x)$', color='red')
plt.plot(x_array, V.detach().numpy(), label=r'Newton-Krylov NN $V(x)$', color='blue')
plt.xlabel(r'$x$')
plt.title(r'Newton-Krylov Equation-Free Steady-State')
plt.legend()

# Average the erros and show convergence
mse = pt.mean(errors, dim=0)
outers = pt.arange(0, outer_iterations+1)
plt.figure()
plt.semilogy(outers, mse, label=r'PDE Error')
rect = mpl.patches.Rectangle((3.5, plt.ylim()[0]), 7.5, plt.ylim()[1], color='gray', alpha=0.2)
ax = plt.gca()
ax.add_patch(rect)
plt.xlabel('# Outer Iterations')
plt.title(r'$M = $' + str(M) + r', $n_{inner} = $' + str(inner_iterations))
plt.legend()
plt.show()
