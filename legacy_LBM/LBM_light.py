import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import matplotlib.pyplot as plt

def LBM(phi_f, phi_g, L, M, dt, Tf, d1, d2, print_dt=None):
	# Method parameters
	kappa_1 = L**2 / d1
	kappa_2 = L**2 / d2
	dx = L / M
	c = dx / dt
	c_s = c / np.sqrt(3.0)
	c_i = np.array([-c, 0.0, c])

	# Setup initial distributions
	weights = np.array([1.0, 4.0, 1.0]) / 6.0
	f_v = np.outer(weights, phi_f)
	g_v = np.outer(weights, phi_g)

	n = 0.0
	if print_dt is not None:
		print_counter = 1

	while n*dt <= Tf:
		# Update phi using finite differences
		phi_f_new = np.copy(phi_f)
		phi_g_new = np.copy(phi_g)
		for j in range(M):
			rhs_f = d1/dx**2*(phi_f[(j-1) % M] - 2.0*phi_f[j] + phi_f[(j+1) % M]) + 1.0 - 2.0*phi_f[j] + phi_f[j]**2 * phi_g[j]
			rhs_g = d2/dx**2*(phi_g[(j-1) % M] - 2.0*phi_g[j] + phi_g[(j+1) % M]) + 3.0                - phi_f[j]**2 * phi_g[j]

			phi_f_new[j] = phi_f[j] + dt*rhs_f
			phi_g_new[j] = phi_g[j] + dt*rhs_g
		phi_f = np.copy(phi_f_new)
		phi_g = np.copy(phi_g_new)

		# Create equilibrium velocity distribution using phi_f,g
		u_f = np.divide(-c*f_v[0,:] + c*f_v[2,:], phi_f)
		u_g = np.divide(-c*g_v[0,:] + c*g_v[2,:], phi_g)
		f_v_eq = np.outer(weights, phi_f)
		g_v_eq = np.outer(weights, phi_g)
		for i in range(3):
			for j in range(M):
				f_v_eq[i,j] = f_v_eq[i,j]*(1.0 + u_f[j]*c_i[i] / c_s**2 + ((u_f[j]*c_i[i])**2 - c_s**2 * u_f[j]**2) / (2.0*c_s**4))
				g_v_eq[i,j] = g_v_eq[i,j]*(1.0 + u_g[j]*c_i[i] / c_s**2 + ((u_g[j]*c_i[i])**2 - c_s**2 * u_g[j]**2) / (2.0*c_s**4))

		# Collision Step
		f_xi = f_v - dt / kappa_1 * (f_v - f_v_eq)
		g_xi = g_v - dt / kappa_2 * (g_v - g_v_eq)

		# Streaming Step with Periodic Boundary Conditions
		for i in range(3):
			diff = i - 1
			for j in range(M):
				f_v[i, (j + diff) % M] = f_xi[i,j]
				g_v[i, (j + diff) % M] = g_xi[i,j]

		# Updating Step
		n += 1.0

		if print_dt is not None and np.abs(n*dt - print_counter*print_dt) < 1.e-8:
			str = 'T = {:.6f} :\t {:.6f} \t {:.6f}'.format(round(n*dt, 6), round(lg.norm(phi_f), 6), round(lg.norm(phi_g), 6))
			print(str)
			print_counter += 1

	return np.sum(f_v, axis=0), np.sum(g_v, axis=0)

def LBM_extended(f_v, g_v, phi_f, phi_g, L, M, dt, Tf, d1, d2, print_dt=None):
	# Method parameters
	kappa_1 = L**2 / d1
	kappa_2 = L**2 / d2
	dx = L / M
	c = dx / dt
	c_s = c / np.sqrt(3.0)
	c_i = np.array([-c, 0.0, c])

	# Setup initial distributions
	weights = np.array([1.0, 4.0, 1.0]) / 6.0
	#phi_f = np.sum(f_v, axis=0)
	#phi_g = np.sum(g_v, axis=0)

	n = 0.0
	if print_dt is not None:
		print_counter = 1

	while n*dt <= Tf:
		# Update phi using finite differences
		phi_f_new = np.copy(phi_f)
		phi_g_new = np.copy(phi_g)
		for j in range(M):
			rhs_f = d1/dx**2*(phi_f[(j-1) % M] - 2.0*phi_f[j] + phi_f[(j+1) % M]) + 1.0 - 2.0*phi_f[j] + phi_f[j]**2 * phi_g[j]
			rhs_g = d2/dx**2*(phi_g[(j-1) % M] - 2.0*phi_g[j] + phi_g[(j+1) % M]) + 3.0                - phi_f[j]**2 * phi_g[j]

			phi_f_new[j] = phi_f[j] + dt*rhs_f
			phi_g_new[j] = phi_g[j] + dt*rhs_g
		phi_f = np.copy(phi_f_new)
		phi_g = np.copy(phi_g_new)

		# Create equilibrium velocity distribution using phi_f,g
		#u_f = np.divide(-c*f_v[0,:] + c*f_v[2,:], phi_f)
		#u_g = np.divide(-c*g_v[0,:] + c*g_v[2,:], phi_g)
		f_v_eq = np.outer(weights, phi_f)
		g_v_eq = np.outer(weights, phi_g)
		#for i in range(3):
		#	for j in range(M):
		#		f_v_eq[i,j] = f_v_eq[i,j]*(1.0 + u_f[j]*c_i[i] / c_s**2 + ((u_f[j]*c_i[i])**2 - c_s**2 * u_f[j]**2) / (2.0*c_s**4))
		#		g_v_eq[i,j] = g_v_eq[i,j]*(1.0 + u_g[j]*c_i[i] / c_s**2 + ((u_g[j]*c_i[i])**2 - c_s**2 * u_g[j]**2) / (2.0*c_s**4))

		# Collision Step
		f_xi = f_v - dt / kappa_1 * (f_v - f_v_eq)
		g_xi = g_v - dt / kappa_2 * (g_v - g_v_eq)

		# Streaming Step with Periodic Boundary Conditions
		for i in range(3):
			diff = i - 1
			for j in range(M):
				f_v[i, (j + diff) % M] = f_xi[i,j]
				g_v[i, (j + diff) % M] = g_xi[i,j]

		# Updating Step
		n += 1.0

		if print_dt is not None and np.abs(n*dt - print_counter*print_dt) < 1.e-8:
			str = 'T = {:.6f} :\t {:.6f} \t {:.6f}'.format(round(n*dt, 6), round(lg.norm(phi_f), 6), round(lg.norm(phi_g), 6))
			print(str)
			print_counter += 1

	return f_v, g_v, phi_f, phi_g

def run_LBM():
	T = 10.0
	M = 200
	L = 1.0
	d1 = 5.e-4
	d2 = 0.06
	dt = 1.e-5
	verbose_dt = 1.e-5

	#filename = 'PDE_steady_state_seed=100_.npy'
	directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
	filename = 'LBM_invariant_dt=1e-05_.npy'
	x_st = np.load(directory+filename).flatten()
	U_st = x_st[0:M]
	V_st = x_st[M:]

	eps = 0.0
	x_array = np.linspace(0.0, 1.0, M)
	U = U_st + eps*np.sin(x_array)
	V = V_st + eps*np.cos(x_array)
	U, V = LBM(U, V, L, M, dt, T, d1, d2, print_dt=verbose_dt)

	# Plot concentrations of U and V
	plt.plot(x_array, U, label=r'$U(x)$', color='red')
	plt.xlabel(r'$x$')
	plt.title(r'Steady-State Lattice-Boltzmann T = ' + str(T))
	plt.plot(x_array, V, label=r'$V(x)$')
	plt.xlabel(r'$x$')

	plt.legend()
	plt.show()

def run_LBM_extended():
	T = 10.0
	M = 200
	L = 1.0
	d1 = 5.e-4
	d2 = 0.06
	dt = 1.e-5
	verbose_dt = 1.e-3
	weights = np.array([1.0, 4.0, 1.0]) / 6.0

	directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
	filename = 'LBM_invariant_dt=1e-05_.npy'
	x_st = np.load(directory+filename).flatten()
	f_v = np.outer(weights, x_st[0:M])
	g_v = np.outer(weights, x_st[M:])

	x_array = np.linspace(0.0, 1.0, M)
	f_v_new, g_v_new = LBM_extended(f_v, g_v, L, M, dt, T, d1, d2, print_dt=verbose_dt)
	phi_f = np.sum(f_v_new, axis=0)
	phi_g = np.sum(g_v_new, axis=0)

	# Plot concentrations of U and V
	plt.plot(x_array, phi_f, label=r'$U(x)$', color='red')
	plt.xlabel(r'$x$')
	plt.title(r'Steady-State Lattice-Boltzmann T = ' + str(T))
	plt.plot(x_array, phi_g, label=r'$V(x)$')
	plt.xlabel(r'$x$')

	plt.legend()
	plt.show()

def run_LBM_interrupted(): # For debugging Only
	T = 10.0
	M = 200
	L = 1.0
	d1 = 5.e-4
	d2 = 0.06
	dt = 1.e-5
	verbose_dt = 1.e-5

	#filename = 'PDE_steady_state_seed=100_.npy'
	directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
	filename = 'LBM_invariant_dt=1e-05_.npy'
	x_st = np.load(directory+filename).flatten()
	U_st = x_st[0:M]
	V_st = x_st[M:]

	eps = 0.0
	x_array = np.linspace(0.0, 1.0, M)
	U = U_st + eps*np.sin(x_array)
	V = V_st + eps*np.cos(x_array)

	N = 100
	dT = T / N
	recorded_U = [np.copy(U)]
	recorded_V = [np.copy(V)]
	for n in range(N):
		print('Interrupted T =', (n+1)*dT)
		U_start = recorded_U[n]
		V_start = recorded_V[n]
		U, V = LBM(np.copy(U_start), np.copy(V_start), L, M, dt, dT, d1, d2, print_dt=verbose_dt)
		recorded_U.append(np.copy(U))
		recorded_V.append(np.copy(V))
	
	for n in range(len(recorded_U)):
		plt.plot(x_array, recorded_U[n], label=r'$T =$' + str(n*dT))
	plt.xlabel(r'$x$')
	plt.title(r'Steady-State Lattice-Boltzmann T = ' + str(T))
	plt.plot(x_array, V, label=r'$V(x)$')
	plt.xlabel(r'$x$')

	plt.legend()
	plt.show()

def run_LBM_extended_interrupted(): # For debugging only
	T = 3.0
	M = 200
	L = 1.0
	d1 = 5.e-4
	d2 = 0.06
	dt = 1.e-5
	verbose_dt = 1.e-3
	weights = np.array([1.0, 4.0, 1.0]) / 6.0

	directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/Fixed_Point_NK_LBM/'
	filename = 'LBM_invariant_dt=1e-05_.npy'
	x_st = np.load(directory+filename).flatten()
	f_v = np.outer(weights, x_st[0:M])
	g_v = np.outer(weights, x_st[M:])
	phi_f = x_st[0:M]
	phi_g = x_st[M:]

	N = 100
	dT = T / N
	recorded_U_f = [np.copy(f_v)]
	recorded_V_f = [np.copy(g_v)]
	recorded_U_phi = [np.copy(phi_f)]
	recorded_V_phi = [np.copy(phi_g)]
	for n in range(N):
		print('Interrupted T =', (n+1)*dT)
		f_v_start = recorded_U_f[n]
		g_v_start = recorded_V_f[n]
		phi_f_start = recorded_U_phi[n]
		phi_g_start = recorded_V_phi[n]
		f_v_new, g_v_new, phi_f_new, phi_g_new = LBM_extended(np.copy(f_v_start), np.copy(g_v_start), np.copy(phi_f_start), np.copy(phi_g_start), L, M, dt, dT, d1, d2, print_dt=verbose_dt)
		recorded_U_f.append(np.copy(f_v_new))
		recorded_V_f.append(np.copy(g_v_new))
		recorded_U_phi.append(phi_f_new)
		recorded_V_phi.append(phi_g_new)
	
	x_array = np.linspace(0.0, 1.0, M)
	for n in range(len(recorded_U_f)):
		plt.plot(x_array, np.sum(recorded_U_f[n], axis=0), label=r'$T =$' + str(n*dT))
	plt.xlabel(r'$x$')
	plt.title(r'Steady-State Lattice-Boltzmann T = ' + str(T))
	plt.xlabel(r'$x$')

	plt.legend()
	plt.show()

if __name__ == '__main__':
	run_LBM_extended_interrupted()