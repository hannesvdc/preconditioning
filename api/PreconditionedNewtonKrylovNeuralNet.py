import autograd.numpy as np
import autograd.numpy.linalg as lg

import api.SuperStructure as ss

# This Newton-Krylov Neural Network only works when F is applied for each column of
# The input matrix.
#
# Internal: Always store extra inputs in third dimension
class ConstantPreconditionedNewtonKrylovNetwork(ss.SuperStructure):
    def __init__(self, F, M, outer_iterations, inner_iterations, baseweight=4.0):
        super().__init__()

        self.eps = 1.e-8

        self.F = F # the function we want to solve, needs to n
        self.dF_v = lambda x, v, Fx: (self.F(x + self.eps*v) - Fx) / self.eps
        self.f = lambda x, v, Fx: self.dF_v(x, v, Fx) + Fx # The linear system during every outer iteration
        self.M = M

        self.matrix_norm_sq = lambda X: np.sum(np.square(X), axis=0)

        self.outer_iterations = outer_iterations
        self.inner_iterations = inner_iterations
        self.baseweight = baseweight

    # 'x' is a matrix with initial vectors in the columns
    def loss(self, x, weights):
        N_data = x.shape[1]
        total_loss = 0.0

        for k in range(self.outer_iterations): # do self.outer_iterations iterations
            loss_weight = self.baseweight**(k+1)
            x = self.inner_forward(x, weights) # x = x_k = solution to self.F(x) = 0 in each column
            total_loss += loss_weight * np.sum(np.square(self.F(x))) # Sum over all data points (axis=1) and over all components (axis=0)

        averaged_loss = total_loss / N_data
        return averaged_loss
    
    def forward(self, x, weights, n_outer_iterations):
        samples = np.zeros((x.shape[0], n_outer_iterations+1, x.shape[1]))
        samples[:,0,:] = x

        for i in range(n_outer_iterations): # do self.outer_iterations iterations
            x = self.inner_forward(x, weights) # x = x_k = solution to self.F(x) = 0; x is a matrix
            samples[:,i+1,:] = x
        return samples
    
    # One complete inner iterations for a matrix of initial conditions
    def inner_forward(self, xk, weights):
        y = np.zeros_like(xk) # y stores the variable that solves f(xk, y) = 0 (i.e. the linear system)
        F_value = self.F(xk) # A matrix with (N, N_data components)
        V = self.f(xk, y, F_value)[:,np.newaxis,:] # v_0 size (N, 1, N_data)

        for n in range(1, self.inner_iterations): # do inner_iterations-1 function evaluations
            yp = self._N(y, V, n, weights)
            _v = self.f(xk, yp, F_value) # v_n is a matrix shape (N, N_data)
            v = lg.solve(self.M, _v)
            V = np.append(V, v[:,np.newaxis,:], axis=1) # v_n size (N, n, N_data)

        yp = self._N(y, V, self.inner_iterations, weights)
        return xk + yp # y = x_{k+1} - x_k
    
    def _N(self, y, V, n, weights):
        lower_index = ( (n-1) * n ) // 2
        upper_index = ( n * (n+1) ) // 2
        return y + np.tensordot(V, weights[lower_index:upper_index], axes=([1],[0]))
    
# This implementation is fully vectorized, computing the N * N * N_data precontiioning matrix is the
# computational bottleneck.
class PrecondtionedNewtonKrylovNetwork(ss.SuperStructure):
    def __init__(self, F, M, outer_iterations, inner_iterations, baseweight=4.0, apply=None):
        super().__init__()

        self.eps = 1.e-8

        self.F = F # the function we want to solve, needs to n
        self.dF_v = lambda x, v, Fx: (self.F(x + self.eps*v) - Fx) / self.eps
        self.f = lambda x, v, Fx: self.dF_v(x, v, Fx) + Fx # The linear system during every outer iteration
        self.M = M
        self.apply = apply

        self.matrix_norm_sq = lambda X: np.sum(np.square(X), axis=0)

        self.outer_iterations = outer_iterations
        self.inner_iterations = inner_iterations
        self.baseweight = baseweight

    def loss(self, x0, weights):
        N_data = x0.shape[1]
        total_loss = 0.0

        max_cond = 0.0
        for n in range(N_data):
            x = x0[:,n]

            for k in range(self.outer_iterations): # do self.outer_iterations iterations
                loss_weight = self.baseweight**(k+1)
                x = self.inner_forward(x, weights) # x = x_k = solution to self.F(x) = 0 in each column
                total_loss += loss_weight * np.sum(np.square(self.F(x))) # Sum over all data points (axis=1) and over all components (axis=0)
                
                try:
                    max_cond = max(max_cond, lg.cond(self.M(x)))
                except:
                    pass

        print('max cond', max_cond)
        averaged_loss = total_loss / N_data
        return averaged_loss
    
    def forward(self, x, weights, n_outer_iterations):
        samples = np.zeros((x.shape[0], n_outer_iterations+1, x.shape[1]))
        samples[:,0,:] = x

        for i in range(n_outer_iterations): # do self.outer_iterations iterations
            x = self.inner_forward(x, weights) # x = x_k = solution to self.F(x) = 0; x is a matrix
            samples[:,i+1,:] = x
        return samples
    
    # One complete inner iterations for a matrix of initial conditions
    def inner_forward(self, xk, weights):
        y = np.zeros_like(xk) # y stores the variable that solves f(xk, y) = 0 (i.e. the linear system)
        F_value = self.F(xk) # A matrix with (N, N_data components)
        V = self.f(xk, y, F_value)[:,np.newaxis] # v_0 size (N, 1)

        # Compute the preconditioning matrix
        J = self.M(xk)

        for n in range(1, self.inner_iterations): # do inner_iterations-1 function evaluations
            yp = self._N(y, V, n, weights)
            _v = self.f(xk, yp, F_value) # v_n is a vector
            v = lg.solve(J, _v)
            V = np.append(V, v[:,np.newaxis], axis=1) # v_n size (N, n)

        yp = self._N(y, V, self.inner_iterations, weights)
        return xk + yp # y = x_{k+1} - x_k
    
    def _N(self, y, V, n, weights):
        lower_index = ( (n-1) * n ) // 2
        upper_index = ( n * (n+1) ) // 2
        return y + np.dot(V, weights[lower_index:upper_index])