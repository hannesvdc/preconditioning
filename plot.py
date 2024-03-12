import autograd.numpy as np
import matplotlib.pyplot as plt

# Model Parameters
M = 200
omega = 100
T = '44.0'

# Load Data
data_directory = '/Users/hannesvdc/Research_Data/Preconditioning_for_Bifurcation_Analysis/'
checkpoint_filename = 'SSA_checkpoint_T=' + T + '_.npy'

checkpoint_data = np.load(data_directory + checkpoint_filename)
U_values = checkpoint_data[0,:]
V_values = checkpoint_data[1,:]

# Plot steady-state concentrations after N updating steps
x_values = np.linspace(0., 1., M)
plt.plot(x_values, U_values/omega, label=r'$U(x)$', color='red')
plt.xlabel(r'$x$')
plt.title(r'$T$ = ' + T)
plt.legend()

plt.figure()
plt.plot(x_values, V_values/omega, label=r'$V(x)$', color='blue')
plt.xlabel(r'$x$')
plt.title(r'$T$ = ' + T)
plt.legend()

plt.show()
