import torch as pt
import torch.nn as nn
from collections import OrderedDict

class InverseJacobianLayer(nn.Module):
    """ Custom Layer that computes J^{-1} rhs using a Krylov Neural-Network """
    def __init__(self, F_macro, inner_iterations):
        super(InverseJacobianLayer, self).__init__()

        self.eps = 1.e-8
        self.F = F_macro
        self.dF_v = lambda xk, w, Fx: (self.F(xk + self.eps*w) - Fx) / self.eps
        self.f = lambda xk, w, Fx, rhs: self.dF_v(xk, w, Fx) - rhs

        self.inner_iterations = inner_iterations
        self.n_weights = (inner_iterations * (inner_iterations + 1) ) // 2
        weights = pt.zeros(self.n_weights)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.

    def forward(self, xk, rhs):
        w = pt.zeros_like(xk)     # w is the solution to J_PDE(xk) w = rhs
        F_value = self.F(xk)      # Stores F(xk), a matrix with (N_data, N) components
        V = self.f(xk, w, F_value,)[:,None,:] # v_0 size (N_data, 1, N)

        for n in range(1, self.inner_iterations): # do inner_iterations-1 function evaluations
            wp = self._N(w, V, n)                 # yp is an (N_data, N) matrix
            v = self.f(xk, wp, F_value, rhs)      # Krylov vectors v is an (N_data, N) matrix
            V = pt.cat((V, v[:,None,:]), dim=1)   # V is an (N_data, n, N) tensor

        wp = self._N(w, V, self.inner_iterations)
        return wp # wp = J_F^{-1} * rhs
    
    def _N(self, w, V, n):
        lower_index = ( (n-1) * n ) // 2
        upper_index = ( n * (n+1) ) // 2
        return w + pt.tensordot(V, self.weights[lower_index:upper_index], dims=([1],[0]))