import torch as pt
import torch.nn as nn

class InverseJacobianLayer(nn.Module):
    """ Custom Layer that computes J^{-1} rhs using a Krylov Neural-Network """
    def __init__(self, F_macro, inner_iterations):
        super(InverseJacobianLayer, self).__init__()

        self.eps = 1.e-8
        self.F = F_macro
        self.dF_w = lambda w, xk, F_value: (self.F(xk + self.eps*w) - F_value) / self.eps
        self.f = lambda w, rhs, xk, F_value: self.dF_w(w, xk, F_value) - rhs

        self.inner_iterations = inner_iterations
        self.n_weights = (inner_iterations * (inner_iterations + 1) ) // 2
        weights = pt.zeros(self.n_weights)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.

    # Input is a tuple containing the nonlinear iterate xk and the right-hand side rhs, 
    # both tensors of shape (N_data, N).
    def forward(self, data):
        xk  = data[0]
        rhs = data[1]
        F_value = self.F(xk)

        # Our initial guess for J_PDE(xk) w = rhs is zeros
        w = pt.zeros_like(xk)
        V = self.f(w, rhs, xk, F_value)[:,None,:] # Krylov vectors with shape (N_data, 1, N)

        # do inner_iterations-1 function evaluations
        for n in range(1, self.inner_iterations):
            wp = self._N(w, V, n)                 # Shape (N_data, N)
            v = self.f(wp, rhs, xk, F_value)
            V = pt.cat((V, v[:,None,:]), dim=1)   # rylov vectors with shape (N_data, 1, N)

        # Linearly combine the Krylov vectors and return the result
        wp = self._N(w, V, self.inner_iterations)
        return wp # wp = J_F^{-1} * rhs
    
    def _N(self, w, V, n):
        lower_index = ( (n-1) * n ) // 2
        upper_index = ( n * (n+1) ) // 2
        return w + pt.tensordot(V, self.weights[lower_index:upper_index], dims=([1],[0]))