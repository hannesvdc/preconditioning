import torch as pt

# Setup the PDE timestepper and psi flowmap function
# Parameters
d1 = 5.e-4; d2 = 0.06
dt = 1.e-4; T = 0.05; N = int(T / dt)
M = 25; dx = 1.0 / M

# Compute indices module M for periodic boundary conditions
def f_vectorized(x):
    U = x[:,0:M]; V = x[:, M:]
    ddU = (pt.roll(U, -1, dims=1) - 2.0*U + pt.roll(U, 1, dims=1)) / dx**2
    ddV = (pt.roll(V, -1, dims=1) - 2.0*V + pt.roll(V, 1, dims=1)) / dx**2
    f1 = d1*ddU + 1.0 - 2.0*U + U**2*V # f1 is a (N_data, M) array
    f2 = d2*ddV + 3.0         - U**2*V # f2 is a (N_data, M) array
    return pt.hstack((f1, f2))

# Apply right-hand side as update (with finite differences)
def PDE_Timestepper_vectorized(x):
	for _ in range(N):
		x = x + dt * f_vectorized(x) # the rhs is an (N_data, 2M) array
	return x
psi = lambda x: PDE_Timestepper_vectorized(x) - x # One-liner