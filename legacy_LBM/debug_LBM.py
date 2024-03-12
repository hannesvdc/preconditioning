import autograd.numpy as np
import autograd.numpy.linalg as lg
import matplotlib.pyplot as plt

import Deterministic_PDE as pde
import LBM as lbm
import LBM_light as lbm_light

T_psi = 0.05
M = 200
L = 1.0
d1 = 5.e-4
d2 = 0.06
dt = 1.e-5

def pde_psi(x):
	x_new = pde.PDE_Timestepper(x, T_psi, M, d2)
	return x - x_new

def lbm_psi(x):
	U = x[0:M]
	V = x[M:]
	U_new, V_new = lbm.LBM(U, V, L, M, dt, T_psi, d1, d2)
	x_new = np.concatenate((U_new, V_new))
	return x - x_new, x_new


def debug_steady_states():
	pde_directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/'
	pde_filename = 'PDE_steady_state_seed=100_.npy'
	pde_x_ss = np.load(pde_directory + pde_filename).flatten()

	lbm_directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
	lbm_filename = 'LBM_Steady_State_Point_T=0.05_d2=0.06_.npy'
	lbm_x_ss = np.load(lbm_directory + lbm_filename).flatten()

	print('SS Absolute Difference', lg.norm(pde_x_ss - lbm_x_ss)/M)

	print('\n\nComputing PDE RHS')
	print('RHS of PDE Solution', lg.norm(pde.f(pde_x_ss, d1, d2)))
	print('RHS of LBM Solution', lg.norm(pde.f(lbm_x_ss, d1, d2)))

	print('\nComputing PDE Psi')
	print('PDE Psi of PDE Solution', lg.norm(pde_psi(pde_x_ss)))
	print('PDE Psi of LBM Solution', lg.norm(pde_psi(lbm_x_ss)))

	print('\nComputing LBM Psi')
	print('LBM Psi of PDE Solution', lg.norm(lbm_psi(pde_x_ss)))
	print('LBM psi of LBM Solution', lg.norm(lbm_psi(lbm_x_ss)))

if __name__ == '__main__':
	debug_steady_states()