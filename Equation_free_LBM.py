import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import matplotlib.pyplot as plt

import LBM_Mesoscopic as lbm

def equation_free(phi_U, phi_V, T_end, T_inner, dT_min, dT_max, dT=None, tolerance=0.1):
	M = phi_U.size
	T = 0.0
	if dT is None:
		dT = max(dT_min, 1.e-3)

	while T <= T_end:
		print('T = ', T, dT)

		# Microscopic Timesteppers
		micro_U, micro_V, micro_U_prev, micro_V_prev, dt = lbm.LBM(np.copy(phi_U), np.copy(phi_V), T_inner, full_output=True)
		d_phi_U = (micro_U - micro_U_prev) / dt
		d_phi_V = (micro_V - micro_V_prev) / dt
		print('Size Derivative', lg.norm(d_phi_U), lg.norm(d_phi_V))

		# Extrapolation
		while True:
			T_outer = T_inner + dT
			phi_U_new = micro_U + (T_outer - T_inner) * d_phi_U
			phi_V_new = micro_V + (T_outer - T_inner) * d_phi_V

			# Do adaptive stepsizing
			if max(lg.norm(phi_U_new - micro_U), lg.norm(phi_V_new - micro_V)) <= M*tolerance:
				phi_U = np.copy(phi_U_new)
				phi_V = np.copy(phi_V_new)
				T += T_outer
				dT = min(1.2 * dT, dT_max)

				break
			else:
				print('Step Size is too Large. Halving Step Size.')
				dT = max(0.5 * dT, dT_min)

	return phi_U, phi_V

def plot_equation_free():
	# Method parameters
	M = 200
	T = 40.0
	T_inner = 0.1
	dT_min = 1.e-8
	dT_max = 1.0
	eq_free_tolerance = 1.e-3

	# Initial Condition
	seed = 100
	rng = rd.RandomState(seed=seed)
	directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
	filename = 'Steady_State_LBM_dt=1e-4.npy'
	x0 = np.load(directory + filename).flatten()

	eps = 3.0
	x_array = np.linspace(0.0, 1.0, M)
	phi_U = x0[0:M] + eps * np.sin(x_array)
	phi_V = x0[M:]  + eps * np.cos(x_array)
	 
	# Run Equation-Free Lattice - Boltzmann
	phi_U, phi_V = equation_free(phi_U, phi_V, T, T_inner, dT_min, dT_max, tolerance=eq_free_tolerance)
	print(phi_U)

	# Plot found solution
	plt.plot(x_array, phi_U, label=r'$U(x)$', color='red')
	plt.plot(x_array, phi_V, label=r'$V(x)$', color='blue')
	plt.xlabel(r'$x$')
	plt.title('Equation-Free Lattice-Boltzmann Steady State')

	plt.legend()
	plt.show()

if __name__ == '__main__':
	plot_equation_free()
