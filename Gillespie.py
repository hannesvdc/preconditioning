import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import matplotlib.pyplot as plt
import scipy.optimize as opt

data_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/'
def SSA(U, V, L, T_end, omega, M, _d1, _d2, checkpoint=False, dT_checkpoint=None, data_dir=None, verbose=False):
	# Setup Model and Reaction Constants
	h = L / M#1.0/M
	k1 = 1.0
	k2 = 2.0
	k3 = 3.0
	k4 = 1.0/omega**2
	d1 = _d1 / h**2
	d2 = _d2 / h**2
	A0 = omega
	B0 = omega

	# Setup Initial Concentrations
	A = A0
	B = B0

	# Do many timesteps to enter steady stat
	T = 0.0
	counter = 0

	while T <= T_end:
		# Calculate propensities and total reaction rate
		propensities = np.array([k1*A*np.ones(M), k2*U, k3*B*np.ones(M), k4*np.multiply(np.multiply(U, U), V),  d1*U, d1*U, d2*V, d2*V])
		propensities = propensities.flatten()
		R = np.sum(propensities)

		# Draw time of next reaction
		u = rd.uniform()
		tau = 1.0/R*np.log(1.0/u)
		T += tau
		if counter % 1000 == 0 and verbose:
			print('T =', T)
		counter += 1

		# Select which reaction happened
		probabilities = propensities / R
		if np.min(probabilities) < 0.0:
			print('Probability Error', probabilities)
			print('A', A)
			print('B', B)
			print('U', np.min(U))
			print('V', np.min(V))
			print('R', R)
			print('Propensities', np.min(propensities))
		k = rd.choice(probabilities.size, p=probabilities)
		reaction_type = int(np.floor((1.0*k) / M))
		box = k - M*reaction_type

		# Carry out the reaction
		if reaction_type == 0:
			U[box] += 1
		elif reaction_type == 1:
			U[box] -= 1
		elif reaction_type == 2:
			V[box] += 1
		elif reaction_type == 3:
			U[box] += 1
			V[box] -= 1
		elif reaction_type == 4:
			U[(box-1) % M] += 1
			U[box] -= 1
		elif reaction_type == 5:
			U[(box+1) % M] += 1
			U[box] -= 1
		elif reaction_type == 6:
			V[(box-1) % M] += 1
			V[box] -= 1
		elif reaction_type == 7:
			V[(box+1) % M] += 1
			V[box] -= 1

	# Make a checkpoint after every second
	if checkpoint and T > T_checkpoint:
		data = np.vstack((U, V))
		print('U', U)
		print('V', V)
		np.save(data_dir + 'SSA_checkpoint_T='+str(T_checkpoint)+'_.npy', data)
		T_checkpoint += dT_checkpoint

	return U, V, counter

def findFixedPointNK():
	# Model Parameters
	omega = 100
	L = 0.2
	M = 40 #200
	T = 1.e-3 #0.05
	d1 = 5.e-4
	d2 = 0.06

	# Define Matrix-Free Helper Functions
	N_SSA = 20
	def psi(x):
		print('Evaluating Psi')
		U = x[0:M]
		V = x[M:]
		x_values = np.zeros((N_SSA, 2*M))
		for n in range(N_SSA):
			U_new, V_new, _ = SSA(np.copy(U), np.copy(V), L, T, omega, M, d1, d2, verbose=False)
			x_new = np.concatenate((U_new, V_new))
			x_values[n,:] = np.copy(x_new)

		x_new = np.average(x_values, axis=0)
		return x - x_new
	
	def d_psi(x, v, f):
		print('Evaluating DPsi')
		e_mach = 1.e-3
		psi_new = psi(x + e_mach*v)

		return (psi_new - f) / e_mach # FD Approximation of Directional Derivative
	
	# Load the original steady-state solution
	print('Loading Steady-State Solution.')
	directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK/'
	x0 = np.load(directory + 'PDE_steady_state_seed=100_.npy').flatten()
	U_st, V_st = x0[0:M], x0[M:]

	# Sample the (random) initial condition
	rng = rd.RandomState()
	eps = 0.01 # 0.0, 0.01 and 0.05 work with PDE steady state
	U = (omega*U_st[0:M]).astype(int)
	V = (omega*V_st[0:M]).astype(int)
	print('V0', V, V_st)
	x0 = np.concatenate((U, V))

	# Run the Newton-Krylov Routine
	tolerance = 1.e-6
	solution = opt.newton_krylov(psi, x0, verbose=True, f_tol=tolerance)
	f_value = psi(solution)
	print('Solution:', solution)
	print('Residue:', f_value, lg.norm(f_value), lg.norm(psi(solution)))

	# Store Found Solution
	directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK/'
	filename = 'PDE_Fixed_Point_T='+str(T)+'_d2='+str(d2)+'_.npy'
	np.save(directory + filename, solution)

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


def plot_SSA_solution():
	# Setup Model Parameters
	omega = 100
	M = 200
	d1 = 5.e-4
	d2 = 0.06

	# Setup Initial Concentrations
	U0 = 2.0
	V0 = 0.75
	seed = 100
	rng = rd.RandomState(seed=seed)
	eps = 0.01
	U = omega*(U0*np.ones(M) + eps*rng.normal(0.0, 1.0, M))
	V = omega*(V0*np.ones(M) + eps*rng.normal(0.0, 1.0, M))
	U = U.astype(int, copy=True)
	V = V.astype(int, copy=True)

	# Setup Method Parameters
	T_end = 100.0
	T_checkpoint = 1.0
	data_dir = data_directory
	U, V = SSA(U, V, T_end, omega, M, d1, d2, checkpoint=True, dT_checkpoint=T_checkpoint, data_dir=data_dir)

	# Plot steady-state concentrations after N updating steps
	x_values = np.linspace(0, 1, M)
	plt.plot(x_values, U, label=r'$U(x)$', color='red')
	plt.legend()

	plt.figure()
	plt.plot(x_values, V, label=r'$V(x)$', color='blue')
	plt.legend()

	plt.show()

if __name__ == '__main__':
	#plot_SSA_solution()
	findFixedPointNK()
