import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from algorithms.NewtonKrylov import *

def Collisions_Reactions(f_1, f0, f1, relaxation_times, A, B, k1, k2, k3, k4, weights, dt):
	phi_U = f_1[0,:] + f0[0,:] + f1[0,:] # Density of U (index 0, all space)
	phi_V = f_1[1,:] + f0[1,:] + f1[1,:] # Density of V (index 0, all space)
	phi = np.vstack((phi_U, phi_V))

	# Collisions, no additonal hermite polynommials needed because u(x, t) = 0 (BGK)
	c_1 = np.zeros_like(phi)
	c0 = np.zeros_like(phi)
	c1 = np.zeros_like(phi)
	for i in range(2): # i ranges of U and V
		c1[i,:]  = -relaxation_times[i] * (f1[i,:]  - weights[2] * phi[i,:]) #phi = rho equiliibrium density
		c0[i,:]  = -relaxation_times[i] * (f0[i,:]  - weights[1] * phi[i,:])
		c_1[i,:] = -relaxation_times[i] * (f_1[i,:] - weights[0] * phi[i,:])

	# Chemical Reactions
	r_1 = np.zeros_like(phi)
	r0 = np.zeros_like(phi)
	r1 = np.zeros_like(phi)
	# reaction term for activator (U)
	propensity_1 = k1*A - k2*phi_U + k4*np.multiply(np.power(phi_U, 2.0), phi_V)
	r1[0,:]  = weights[2] * dt * propensity_1
	r0[0,:]  = weights[1] * dt * propensity_1
	r_1[0,:] = weights[0] * dt * propensity_1
	# reaction term for inibitor (V)
	propensity_2 = k3*B - k4*np.multiply(np.power(phi_U, 2.0), phi_V)
	r1[1,:]  = weights[2] * dt * propensity_2
	r0[1,:]  = weights[1] * dt * propensity_2
	r_1[1,:] = weights[0] * dt * propensity_2

	return c_1, c0, c1, r_1, r0, r1

def _LatticeBM_Schnakenberg(f_1, f0, f1, relaxation_times, A, B, k1, k2, k3, k4, weights, dt):
	M = f0.shape[1]

	# Streaming in intermediate variable
	c_1, c0, c1, r_1, r0, r1 = Collisions_Reactions(f_1, f0, f1, relaxation_times, A, B, k1, k2, k3, k4, weights, dt)
	f1star  = f1  + c1 + r1
	f0star  = f0  + c0 + r0
	f_1star = f_1 + c_1 + r_1
	
	# Updating, use periodc boundary conditions
	for j in range(M):
		f0[:,j] = f0star[:,j]
		f1[: ,(j+1) % M] = f1star[:,j]
		f_1[:, (j-1) % M] = f_1star[:,j]
   
	return f_1, f0, f1

# Implements D1Q3 Lattice-Boltzmann
def LBM(U, V, T, verbose=False, full_output=False):
	# Lattice Parameters
	M = 200
	L = 1.0
	dx = L/M

	# Chemical Model Parameters
	A = 1; B = 1; k1 = 1.0; k2 = 2.0; k3 = 3.0; k4 = 1.0 
	d1 = 5.e-4
	d2 = 0.06

	# LBM Method parameters
	dt = 1.e-4
	weights = np.array([1.0, 4.0, 1.0]) / 6.0 # D1Q3 weights
	cs_quad = weights[0] + weights[2] #speed of sound squared
	relaxation_times = np.array([2.0/(1.0 + 2.0/cs_quad*d1*dt/dx**2), 2.0/(1.0 + 2.0/cs_quad*d2*dt/dx**2)])

	# Initial Condition for Lattice-Boltzmann. Each f is 2 by M
	f_1 = weights[0] * np.vstack((U, V)) # Move to the left
	f0  = weights[1] * np.vstack((U, V)) # Stay Put
	f1  = weights[2] * np.vstack((U, V)) # Move to the right

	if full_output:
		phi_U_history = []
		phi_V_history = []
		t_history = []

	# Timestepping
	n = 0
	n_steps = T / dt 
	while n < n_steps:
		if verbose and n % 1000 == 0:
			print('T = ', n*dt)
		f_1_new, f0_new, f1_new = _LatticeBM_Schnakenberg(np.copy(f_1), 
														 np.copy(f0), 
														 np.copy(f1), 
														 relaxation_times,
														 A, B, k1, k2, k3, k4, 
														 weights, dt)
		f_1 = np.copy(f_1_new)
		f0 = np.copy(f0_new)
		f1 = np.copy(f1_new)
		n += 1

		if full_output:
			phi_U = f_1[0,:] + f0[0,:] + f1[0,:] # Density of U (index 0, all space)
			phi_V = f_1[1,:] + f0[1,:] + f1[1,:] # Density of V (index 0, all space)
			phi_U_history.append(phi_U)
			phi_V_history.append(phi_V)
			t_history.append((n+1.0) * dt)

	# Comppute the densities for plotting
	phi_U = f_1[0,:] + f0[0,:] + f1[0,:] # Density of U (index 0, all space)
	phi_V = f_1[1,:] + f0[1,:] + f1[1,:] # Density of V (index 0, all space)

	if full_output:
		return phi_U, phi_V, phi_U_history[-6], phi_V_history[-6], 5*dt
	else:
		return phi_U, phi_V

def plot_LBM():
	# Method parameters
	M = 200
	T = 100.0

	# Initial Condition
	seed = 100
	rng = rd.RandomState(seed=seed)
	eps = 0.01
	U0 = 2.0
	V0 = 0.75
	U = U0*np.ones(M) + eps * rng.normal(0.0, 1.0, M)
	V = V0*np.ones(M) + eps * rng.normal(0.0, 1.0, M)

	# Run Lattice - Boltzmann
	phi_U, phi_V = LBM(U, V, T, verbose=True)

	# Storing the steady-state for later use
	directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
	filename = 'Steady_State_LBM_dt=1e-4.npy'
	#np.save(directory + filename, np.vstack((phi_U, phi_V)))

	# Plot found solution
	x_array = np.linspace(0.0, 1.0, M)
	plt.plot(x_array, phi_U, label=r'$U(x)$', color='red')
	plt.plot(x_array, phi_V, label=r'$V(x)$', color='blue')
	plt.xlabel(r'$x$')
	plt.title('Lattice-Boltzmann Steady State')

	plt.legend()
	plt.show()

def NewtonKrylovLBM(store=False):
	M = 200
	T_psi = 0.5

	# Load the initial condition (steady-state for now)
	eps = 0.1
	rng = rd.RandomState()
	directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
	filename = 'Steady_State_LBM_dt=1e-4.npy'
	x0 = np.load(directory + filename).flatten()
	x0 = x0 + eps*rng.normal(0.0, 1.0, x0.size)
	
	# psi function
	def psi(x):
		print('Evaluating Psi')
		U = x[0:M]; V = x[M:]
		U_new, V_new = LBM(U, V, T_psi)
		x_new = np.concatenate((U_new, V_new))

		return x - x_new
	
	# Parameters for the Newton-Krylov Method
	tolerance = 1.e-12
	solution = opt.newton_krylov(psi, x0, rdiff=1.e-8, verbose=True, f_tol=tolerance)
	phi_U = solution[0:M]; phi_V = solution[M:]
	print('Solution:', solution)
	print('Residue:', lg.norm(psi(solution)))

	# Store solution
	if store:
		filename = 'Fixed_Point_NK_LBM_initial=steadystate.npy'
		np.save(directory+filename, solution)

	# Plot the fixed point
	x_array = np.linspace(0.0, 1.0, M)
	plt.plot(x_array, phi_U, label=r'$U(x)$', color='red')
	plt.plot(x_array, phi_V, label=r'$V(x)$', color='blue')
	plt.xlabel(r'$x$')
	plt.title('NK-LBM Fixed Point')

	plt.legend()
	plt.show()


if __name__ == '__main__':
	plot_LBM()
	#NewtonKrylovLBM()