import sys
sys.path.append('../')

import autograd.numpy as np
import autograd.numpy.linalg as lg
import matplotlib.pyplot as plt

import LBM_Mesoscopic as lbm

# This routine tests whether psi is well-defined
def testLBMIncrements():
	# Model and Method parameters
	M = 200
	T_f = 1.0
	N_points = 100
	T_range = np.linspace(0.0, T_f, N_points+1)

	# Load the initial condition
	directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
	filename = 'Steady_State_LBM_dt=1e-4.npy'
	data = np.load(directory + filename)
	initial_U = data[0,:]
	initial_V = data[1,:]

	phi_U_values = []
	phi_V_values = []
	for n in range(len(T_range)):
		print('T =', T_range[n])

		phi_U, phi_V = lbm.LBM(np.copy(initial_U), np.copy(initial_V), T_range[n], verbose=False)
		phi_U_values.append(phi_U)
		phi_V_values.append(phi_V)

	x_range = np.linspace(0.0, 1.0, M)
	for n in range(len(T_range)):
		plt.plot(x_range, phi_U_values[n])
		plt.plot(x_range, phi_V_values[n])

	plt.show()

def testLBMPsi():
	M = 200
	T_f = 1.0
	N_points = 100
	T_range = np.linspace(0.0, T_f, N_points+1)

	def psi(x, T):
		U = x[0:M]; V = x[M:]
		U_new, V_new = lbm.LBM(U, V, T, verbose=False)
		return lg.norm(x - np.concatenate((U_new, V_new)))
	
	# Load the initial condition
	directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
	filename = 'Steady_State_LBM_dt=1e-4.npy'
	initial = np.load(directory + filename).flatten()

	psi_values = []
	for n in range(len(T_range)):
		print('T =', T_range[n])
		psi_values.append(psi(initial, T_range[n]))

	plt.plot(T_range, psi_values, label=r'$\psi(T)$')
	plt.xlabel(r'$T$')
	plt.ylabel(r'$\psi$', rotation=0)
	plt.legend()
	plt.show()

if __name__ == '__main__':
	#testLBMIncrements()
	testLBMPsi()