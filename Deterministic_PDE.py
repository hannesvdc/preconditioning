import autograd.numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import numpy as onp

from api.algorithms.NewtonKrylov import *

# x contains samples along the columns (axis = 1)
def f_vectorized(x, d1, d2, M):
	dx = 1.0/M
	U = x[0:M, :]
	V = x[M:, :]

	# Compute indices module M for periodic boundary conditions
	ddU = (np.roll(U, -1, axis=0) - 2.0*U + np.roll(U, 1, axis=0)) / dx**2
	ddV = (np.roll(V, -1, axis=0) - 2.0*V + np.roll(V, 1, axis=0)) / dx**2
	f1 = d1*ddU + 1.0 - 2.0*U + U**2*V # f1 is a (M, N_data) array
	f2 = d2*ddV + 3.0         - U**2*V # f2 is a (M, N_data) array

	return np.vstack((f1, f2))

def f_fast(x, d1, d2, M):
	dx = 1.0/M
	U = x[0:M]
	V = x[M:]

	# Compute indices module M for periodic boundary conditions
	ddU = (np.roll(U, -1) - 2.0*U + np.roll(U, 1)) / dx**2
	ddV = (np.roll(V, -1) - 2.0*V + np.roll(V, 1)) / dx**2
	f1 = d1*ddU + 1.0 - 2.0*U + U**2*V
	f2 = d2*ddV + 3.0         - U**2*V

	return np.concatenate((f1, f2))

def PDE_Timestepper_vectorized(x, parameters, verbose=False):
	M = parameters['M']
	T = parameters['T']
	d2 = parameters['d2']
	d1 = 5.e-4
	dt = 1.e-4
	N = int(T / dt)

	if verbose:
		print('U', x[0:M, :])
		print('V', x[M:, :])

	for n in range(N):
		# Apply right-hand side as update (with finite differences)
		f_rhs = f_vectorized(x, d1, d2, M) # f_rhs is an (2M, N_data) array
		x = x + dt*f_rhs

		# Update timestepping
		if verbose and n % 1000 == 0:
			print('T =', n*dt)
			print('U', x[0:M, :])

	return x

def PDE_Timestepper(x, parameters, verbose=False):
	M = parameters['M']
	T = parameters['T']
	d2 = parameters['d2']
	d1 = 5.e-4
	dt = 1.e-4
	N = int(T / dt)

	if verbose:
		print('U', x[0:M])
		print('V', x[M:])

	for n in range(N):
		# Apply right-hand side as update (with finite differences)
		f_rhs = f_fast(x, d1, d2, M)
		x = x + dt*f_rhs

		# Update timestepping
		if verbose and n % 1000 == 0:
			print('T =', n*dt)
			print('U', x[0:M])

	return x

def findFixedPointNK():
	# Model Parameters
	M = 200
	T = 0.05
	d1 = 5.e-4
	d2 = 0.06
	parameters = {'M': M, 'T': T, 'd2': d2}

	# Define Matrix-Free Helper Functions
	def psi(x):
		print('Evaluating Psi')
		x_new = PDE_Timestepper(x, parameters)

		return x - x_new
	
	def d_psi(x, v, f):
		print('Evaluating DPsi')
		e_mach = 1.e-8
		psi_new = psi(x + e_mach*v)

		return (psi_new - f) / e_mach # FD Approximation of Directional Derivative
	
	# Load the original steady-state solution
	print('Loading Steady-State Solution.')
	directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK/'
	x0 = onp.load(directory + 'PDE_steady_state_seed=100_.npy').flatten()
	U_st, V_st = x0[0:M], x0[M:]

	# Sample the (random) initial condition
	rng = rd.RandomState()
	eps = 0.01 # 0.0, 0.01 and 0.05 work with PDE steady state
	#U = U_st + eps*rng.normal(0.0, 1.0, M)
	#V = V_st + eps*rng.normal(0.0, 1.0, M)
	U = 2.0*np.ones(M)  + eps*rng.normal(0.0, 1.0, M)
	V = 0.75*np.ones(M) + eps*rng.normal(0.0, 1.0, M)
	x0 = np.concatenate((U, V))
	
	# Run the Newton-Krylov Routine
	max_it = 100
	tolerance = 1.e-6
	solution, f_value = NewtonKrylov(psi, d_psi, x0, max_it, tolerance=tolerance, verbose=True)
	print('Solution:', solution)
	print('Residue:', f_value, lg.norm(f_value), lg.norm(f_fast(solution, d1, d2, M)))

	# Store Found Solution
	directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK/'
	filename = 'PDE_Fixed_Point_T='+str(T)+'_d2='+str(d2)+'_.npy'
	onp.save(directory + filename, solution)

	# Plot found solution
	U = solution[0:M]
	V = solution[M:]
	x_array = np.linspace(0.0, 1.0, M)
	plt.plot(x_array, U, label=r'$U(x)$', color='red')
	plt.xlabel(r'$x$')
	plt.legend()

	plt.figure()
	plt.plot(x_array, V, label=r'$V(x)$')
	plt.xlabel(r'$x$')

	plt.legend()
	plt.show()


def Plot_PDE_Solution():
	seed = 100
	rng = rd.RandomState(seed=seed)

	T = 100.0
	M = 200
	d2 = 0.06
	parameters = {'M': M, 'T': T, 'd2': d2}

	eps = 0.01
	U = 2.0*np.ones(M)  + eps*rng.normal(0.0, 1.0, M)
	V = 0.75*np.ones(M) + eps*rng.normal(0.0, 1.0, M)
	x = np.concatenate((U, V))

	x = PDE_Timestepper(x, parameters, verbose=True)
	U = x[0:M]
	V = x[M:]

	T_psi = 0.05
	psi_parameters = {'M': M, 'T': T_psi, 'd2': d2}
	def psi(x):
		print('Evaluating Psi')
		x_new = PDE_Timestepper(x, psi_parameters)

		return x - x_new
	print('psi PDE', lg.norm(psi(np.concatenate((U, V)))))
	
	# Storing Result for future reference
	directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/'
	filename = 'PDE_steady_state_seed='+str(seed)+'_.npy'
	np.save(directory +  filename, np.vstack((U,V)))

	# Plot concentrations of U and V
	x_array = np.linspace(0.0, 1.0, M)[::4]
	plt.plot(x_array, U[::4], label=r'$U(x)$', color='red')
	plt.xlabel(r'$x$')
	plt.title(r'Steady-State PDE')
	plt.legend()

	plt.figure()
	plt.plot(x_array, V[::4], label=r'$V(x)$')
	plt.title(r'Steady-State PDE')
	plt.xlabel(r'$x$')

	plt.legend()
	plt.show()

if __name__ == '__main__':
	#findFixedPointNK()
	Plot_PDE_Solution()
