import sys
sys.path.append('../')

import torch as pt
import numpy as np

import ChemicalRoutines as cr
from api.PreconditionedNewtonKrylovImpl import *

pt.set_default_dtype(pt.float64)
pt.set_grad_enabled(True)

# Setup the reference function
T_psi = 0.05
F = lambda x: cr.psi_pde(x, T_psi)
F_marginal = lambda x: F(pt.unsqueeze(x, dim=0))[0]

# Load the point of interest
M = 200
ss_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
ss_filename = 'Steady_State_LBM_dt=1e-4.npy'
x_ss = pt.from_numpy(np.load(ss_directory + ss_filename).flatten()).requires_grad_()
w = pt.randn(size=(2*M,)).requires_grad_()

# Compute the analytical Jacobian
unit_vectors = pt.eye(2 * M)
def dF(xp):
    jacobian_rows = [pt.autograd.grad(F_marginal(xp), xp, vec)[0] for vec in unit_vectors]
    return pt.stack(jacobian_rows)

# Compute analytical Jacobian-vector inproducts
jacobian = dF(x_ss)
jacobian_w = pt.matmul(jacobian, w)
print('Steady-State Jacobian Condition Number:', pt.linalg.cond(jacobian))

# Compute the numerical jacobian-vector inproduct
layer = InverseJacobianLayer(F, 4)
layer.computeFValue(pt.unsqueeze(x_ss, dim=0))
dF_w = layer.dF_w(pt.unsqueeze(w, dim=0), pt.unsqueeze(x_ss, dim=0))[0]

print('Finite Differences Relative Error', pt.norm(dF_w - jacobian_w) / pt.norm(jacobian_w))