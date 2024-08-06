import torch as pt
import torch.nn as nn
from collections import OrderedDict

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
    def forward(self, x):
        xk  = x[0]
        rhs = x[1]
        w   = x[2]

        F_value = self.F(xk)
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
    
class InverseJacobianNetwork(nn.Module):
    """ Custom Neural Network that computes J^{-1} rhs using an 
        InverseJacobianLayer. """
    def __init__(self, F_macro, inner_iterations):
        super(InverseJacobianNetwork, self).__init__()

        self.inv_jac_layer = InverseJacobianLayer(F_macro, inner_iterations)
        layer_list = [('layer_0', self.inv_jac_layer)]
        self.layers = nn.Sequential(OrderedDict(layer_list))

        # Check the number of preconditioning parameters
        print('Number of Preconditioning Parameters:', sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        return self.layers(x)
    
class PreconditionedNewtonKrylovLayer(nn.Module):
    """ Custom Preconditioned Newton-Krylov layer to solve M * (JF(xk) y = -F(xk)) """
    def __init__(self, F, inner_iterations, inv_jac_network):
        super(PreconditionedNewtonKrylovLayer, self).__init__()

        self.eps = 1.e-8
        self.F = F # the function we want to solve, needs to n
        self.dF_v = lambda x, v, Fx: (self.F(x + self.eps*v) - Fx) / self.eps
        self.f = lambda x, v, Fx: self.dF_v(x, v, Fx) + Fx # The linear system during every outer iteration
        
        self.inner_iterations = inner_iterations
        self.n_weights = (inner_iterations * (inner_iterations + 1) ) // 2
        weights = pt.zeros(self.n_weights)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.

        self.inv_jac_network = inv_jac_network

    def forward(self, xk):
        y = pt.zeros_like(xk)     # y stores the variable that solves f(xk, y) = 0 (i.e. the linear system)
        F_value = self.F(xk)      # A matrix with (N_data, N) components
        v0 = self.f(xk, y, F_value) # v_0 size (N_data, 1, N)
        Mv0 = self.inv_jac_network((xk, v0, pt.zeros_like(xk))) # Preconditioning
        V = Mv0[:,None,:]
        # TODO Merge this double code in one for-loop
        for n in range(1, self.inner_iterations): # do inner_iterations-1 function evaluations
            yp = self._N(y, V, n)                 # yp is an (N_data, N) matrix
            v  = self.f(xk, yp, F_value)           # v is an (N_data, N) matrix
            Mv = self.inv_jac_network((xk, v, pt.zeros_like(xk))) # Preconditioning
            V = pt.cat((V, Mv[:,None,:]), dim=1)   # V is an (N_data, n, N) tensor

        yp = self._N(y, V, self.inner_iterations)
        return xk + yp # y = x_{k+1} - x_k
    
    def _N(self, y, V, n):
        lower_index = ( (n-1) * n ) // 2
        upper_index = ( n * (n+1) ) // 2
        return y + pt.tensordot(V, self.weights[lower_index:upper_index], dims=([1],[0]))

class PreconditionedNewtonKrylovNetwork(nn.Module):
    def __init__(self, F, F_macro, inner_iterations : tuple[int, int]):
        super(PreconditionedNewtonKrylovNetwork, self).__init__()
        
        # Setup the InverseJacobianNetwork and the inner layer
        self.inv_jac_network = InverseJacobianNetwork(F_macro, inner_iterations[1])
        self.inner_layer = PreconditionedNewtonKrylovLayer(F, inner_iterations[0], self.inv_jac_network)

        # Include layer parameters as trainable parameters
        self.params = []
        self.params.extend(self.layers.parameters())
        self.params.extend(self.inv_jac_network.parameters())

        # Check the total number of parameters
        print('Total Number of Preconditioned Newton-Krylov Parameters:', sum(p.numel() for p in self.parameters()))

    # Input: Nonlinear iterate xk after k outer iterations
    def forward(self, xk):
        return self.inner_layer(xk)

class PreconditionedNewtonKrylovLoss(nn.Module):
    def __init__(self, network : PreconditionedNewtonKrylovNetwork, 
                       F, 
                       outer_iterations : int, 
                       base_weight=4.0):
        super(PreconditionedNewtonKrylovLoss, self).__init__()
        
        self.F = F
        self.network = network
        self.outer_iterations = outer_iterations
        self.base_weight = base_weight

    def forward(self, x):
        loss = 0.0
        N_data = x.shape[0]

        for k in range(self.outer_iterations):
            x = self.network.forward(x)

            loss_weight = self.base_weight**(k+1)
            loss += loss_weight * pt.sum(pt.square(self.F(x))) # Sum over all data points (dim=0) and over all components (dim=1)

        avg_loss = loss / N_data
        return avg_loss