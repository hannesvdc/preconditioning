import sys
sys.path.append('../')

import autograd.numpy as np
import autograd.numpy.linalg as lg
import numpy.random as rd
import matplotlib.pyplot as plt
import scipy.optimize as opt

from NewtonKrylov import *
from LBM_light import *

def newton_krylov_LBM():
	# Model Parameters
	L = 1.0
	M = 200
	T = 0.05
	d1 = 5.e-4
	d2 = 0.06
	dt = 1.e-5

	# Define Matrix-Free Helper Functions
	def psi(x):
		print('Evaluating Psi')
		U = x[0:M]
		V = x[M:]
		U_new, V_new = LBM(U, V, L, M, dt, T, d1, d2)
		x_new = np.concatenate((U_new, V_new))
		return x - x_new
	
	def d_psi(x, v, f):
		print('Evaluating DPsi')
		e_mach = 1.e-8
		psi_new = psi(x + e_mach*v)

		return (psi_new - f) / e_mach # FD Approximation of Directional Derivative
	
	# Load the original steady-state solution
	print('Loading Steady-State Solution.')
	directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
	filename = 'LBM_invariant_dt=1e-05_.npy'
	x0 = np.load(directory + filename).flatten()
	U_st, V_st = x0[0:M], x0[M:]
	print('U_st', lg.norm(psi(np.concatenate((U_st, V_st)))), U_st)

	# Sample the (random) initial condition
	eps = 0.0
	x_array = np.linspace(0.0, 1.0, M)
	U = U_st + eps*np.sin(x_array)
	V = V_st + eps*np.cos(x_array)
	x0 = np.concatenate((U, V))
	
	# Run the Newton-Krylov Routine
	max_it = 100
	tolerance = 1.e-4
	#solution, f_value = NewtonKrylov(psi, d_psi, x0, max_it, tolerance=tolerance, verbose=True, cb_type='x')
	def cb(x, f):
		print('current iterate', lg.norm(f), lg.norm(x))
	solution = opt.newton_krylov(psi, x0, verbose=True, callback=cb, f_tol=tolerance)
	print('Solution:', solution)
	print('Residue:', lg.norm(psi(solution)))

	# Store Found Solution
	directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
	filename = 'LBM_NK_Fixed_Point_T='+str(T)+'_d2='+str(d2)+'_.npy'
	#np.save(directory + filename, solution)

	# Plot found solution
	U = solution[0:M]
	V = solution[M:]
	plt.plot(x_array, U, label=r'$U(x)$', color='red')
	plt.xlabel(r'$x$')
	plt.ylim((0.0, 6.0))
	plt.title(r'Fixed Point Lattice-Boltzmann')
	plt.legend()

	plt.figure()
	plt.plot(x_array, V, label=r'$V(x)$')
	plt.ylim((0.0, 2.0))
	plt.title(r'Fixed Point Lattice-Boltzmann')
	plt.xlabel(r'$x$')

	plt.legend()
	plt.show()

def newton_krylov_LBM_extended():
	# Model Parameters
	L = 1.0
	M = 200
	T = 0.05
	d1 = 5.e-4
	d2 = 0.06
	dt = 1.e-5
	weights = np.array([1.0, 4.0, 1.0]) / 6.0

	# Define Matrix-Free Helper Functions
	def psi(x): # x is in R^{6m}
		print('Evaluating Psi')
		f_v = np.array([x[0:M], x[M:2*M], x[2*M:3*M]])
		g_v = np.array([x[3*M:4*M], x[4*M:5*M], x[5*M:6*M]])
		phi_f = np.sum(f_v, axis=0)
		phi_g = np.sum(g_v, axis=0)

		f_v_new, g_v_new, _, _ = LBM_extended(f_v, g_v, phi_f,phi_g, L, M, dt, T, d1, d2)
		x_new = np.concatenate((f_v_new.flatten(), g_v_new.flatten()))
		return x - x_new
	
	# Load the original steady-state solution
	print('Loading Steady-State Solution.')
	directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
	filename = 'LBM_invariant_dt=1e-05_.npy'
	x0 = np.load(directory + filename).flatten()
	U_st, V_st = x0[0:M], x0[M:]
	f_v = np.outer(weights, U_st)
	g_v = np.outer(weights, V_st)
	x0 = np.concatenate((f_v.flatten(), g_v.flatten()))
	print('U_st', lg.norm(psi(x0)), U_st)
	
	# Run the Newton-Krylov Routine
	tolerance = 2.e-5
	def cb(x, f):
		print('shape', x.shape)
		print('current iterate', lg.norm(f), lg.norm(x))
	solution = opt.newton_krylov(psi, x0, verbose=True, callback=cb, f_tol=tolerance)
	print('Solution:', solution)
	print('Residue:', lg.norm(psi(solution)))

	# Plot the Solution
	f_sol = np.array([solution[0:M], solution[M:2*M], solution[2*M:3*M]])
	g_sol = np.array([solution[3*M:4*M], solution[4*M:5*M], solution[5*M:6*M]])
	phi_f = np.sum(f_sol, axis=0)
	phi_g = np.sum(g_sol, axis=0)

	x_array = np.linspace(0.0, 1.0, M)
	plt.plot(x_array, phi_f, label=r'$U(x)$', color='red')
	plt.xlabel(r'$x$')
	plt.ylim((0.0, 6.0))
	plt.title(r'Fixed Point Lattice-Boltzmann')
	plt.legend()

	plt.figure()
	plt.plot(x_array, phi_g, label=r'$V(x)$')
	plt.ylim((0.0, 2.0))
	plt.title(r'Fixed Point Lattice-Boltzmann')
	plt.xlabel(r'$x$')

	plt.legend()
	plt.show()

if __name__ == '__main__':
	newton_krylov_LBM_extended()