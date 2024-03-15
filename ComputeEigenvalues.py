"""
The goal of this file is to recreate Figure 6 in "Spatially Distributed Stochastic Systems: 
equation-free and equation-assisted preconditioned computation". This code only uses the three 
different numerical integrators (PDE finite differences, microscopic Lattice-Boltzmann and
the equation-free method based on the microscopic Lattice-Boltzmann method). Once this figure
have been reproduced, the code in this file serves no further purpose and should be left alone.

"""

import autograd.numpy as np
import scipy.linalg as lg
import matplotlib.pyplot as plt

import Deterministic_PDE as pde
import Equation_free_LBM as eq_free

T = 0.5
M = 200
d2 = 0.06

T_inner = 0.1
dT_min = 1.e-8
dT_max = 0.4
tolerance = 1.e-4

def PDE_psi(x):
	x_new = pde.PDE_Timestepper(x, T, M, d2)
	return x - x_new

def d_PDE_psi(x, v, f):
	print('Computing PDE dPsi')
	e_mach = 1.e-8
	psi_new = PDE_psi(x + e_mach*v)
	return (psi_new - f) / e_mach # FD Approximation of Directional Derivative

def EQF_psi(x):
	U = x[0:M]; V = x[M:]
	U, V = eq_free.equation_free(U, V, T, T_inner, dT_min, dT_max, dT=None, tolerance=tolerance, verbose=False)
	return x - np.concatenate((U, V))

def d_EQF_psi(x, v, f):
	print('Computing EQF dPsi')
	e_mach = 1.e-8
	psi_new = EQF_psi(x + e_mach*v)
	return (psi_new - f) / e_mach # FD Approximation of Directional Derivative

def PDE_Eigenvalues():
	point = 'stable'
	if point == 'unstable':
		x = np.concatenate((2.0*np.ones(M), 0.75*np.ones(M)))
	else:
		directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK/'
		x = np.load(directory + 'PDE_steady_state_seed=100_.npy').flatten()
	f = PDE_psi(x)
	dpsi_v = lambda v: d_PDE_psi(x, v, f)

	# Create Jacoobian Matrix
	A = np.eye(x.size)
	for i in range(x.size):
		print('i =', i)
		A[:,i] = dpsi_v(A[:,i]) # A[:,i] is initially the i-th unit vector
	eig_vals = lg.eigvals(A)

	# Plot the eigenvalues in the complex plane
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None', linestyle='-')
	ax.add_patch(circ)
	plt.axhline(y=0, color='gray')
	plt.axvline(x=0, color='gray')
	for n in range(eig_vals.size):
		plt.plot(np.real(eig_vals[n]), np.imag(eig_vals[n]), marker='x', color='k')
	plt.xlim((-0.28, 1.05))
	plt.ylim((-0.48, 0.48))
	plt.title('Eigenvalues of PDE Timestepper')
	plt.show()

def EQF_LBM_Eigenvalues():
	point = 'stable'
	if point == 'unstable':
		x = np.concatenate((2.0*np.ones(M), 0.75*np.ones(M)))
	else:
		directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
		filename = 'Steady_State_LBM_dt=1e-4.npy'
		x = np.load(directory + filename).flatten()
	f = EQF_psi(x)
	dpsi_v = lambda v: d_EQF_psi(x, v, f)

	# Create Jacobian Matrix
	A = np.eye(x.size)
	for i in range(x.size):
		print('i =', i)
		A[:,i] = dpsi_v(A[:,i]) # A[:,i] is initially the i-th unit vector
	eig_vals = lg.eigvals(A)

	# Plot the eigenvalues in the complex plane
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None', linestyle='-')
	ax.add_patch(circ)
	plt.axhline(y=0, color='gray')
	plt.axvline(x=0, color='gray')
	for n in range(eig_vals.size):
		plt.plot(np.real(eig_vals[n]), np.imag(eig_vals[n]), marker='x', color='k')
	plt.xlim((-0.28, 1.05))
	plt.ylim((-0.48, 0.48))
	plt.title('Eigenvalues of Equation-Free Coarse Timestepper')
	plt.show()

def Preconditioned_Eigenvalues():
	M  = 200
	point = 'unstable'
	if point == 'unstable':
		x = np.concatenate((2.0*np.ones(M), 0.75*np.ones(M)))
	else:
		directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
		filename = 'Steady_State_LBM_dt=1e-4.npy'
		x = np.load(directory + filename).flatten()
	f_eqf = EQF_psi(x)
	f_pde = PDE_psi(x)
	dpsi_v_eqf = lambda v: d_EQF_psi(x, v, f_eqf)
	dpsi_v_pde = lambda v: d_PDE_psi(x, v, f_pde)

	A_eqf = np.eye(x.size)
	A_pde = np.eye(x.size)
	for i in range(x.size):
		print('i =', i)
		A_eqf[:,i] = dpsi_v_eqf(A_eqf[:,i])
		A_pde[:,i] = dpsi_v_pde(A_pde[:,i])
	M = np.matmul(lg.inv(A_pde), A_eqf)
	eig_vals = lg.eigvals(M)

	# Plot the eigenvalues in the complex plane
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None', linestyle='-')
	ax.add_patch(circ)
	plt.axhline(y=0, color='gray')
	plt.axvline(x=0, color='gray')
	for n in range(eig_vals.size):
		plt.plot(np.real(eig_vals[n]), np.imag(eig_vals[n]), marker='x', color='k')
	plt.title('Eigenvalues of Preconditioned Jaoobian')
	plt.xlabel(r'Re $\lambda$')
	plt.ylabel(r'Im $\lambda$')
	plt.show()

if __name__ == '__main__':
	Preconditioned_Eigenvalues()