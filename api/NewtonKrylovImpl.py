import torch as pt
import torch.nn as nn
from collections import OrderedDict

class NewtonKrylovLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, F, inner_iterations):
        super(NewtonKrylovLayer, self).__init__()

        self.eps = 1.e-8
        self.F = F # the function we want to solve, needs to n
        self.dF_v = lambda x, v, Fx: (self.F(x + self.eps*v) - Fx) / self.eps
        self.f = lambda x, v, Fx: self.dF_v(x, v, Fx) + Fx # The linear system during every outer iteration
        
        self.inner_iterations = inner_iterations
        self.n_weights = (inner_iterations * (inner_iterations + 1) ) // 2
        weights = pt.zeros(self.n_weights)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.

    def forward(self, x):
        y = pt.zeros_like(x)     # y stores the variable that solves f(xk, y) = 0 (i.e. the linear system)
        F_value = self.F(x)      # A matrix with (N_data, N) components
        V = self.f(x, y, F_value)[:,None,:] # v_0 size (N_data, 1, N)

        for n in range(1, self.inner_iterations): # do inner_iterations-1 function evaluations
            yp = self._N(y, V, n)                 # yp is an (N_data, N) matrix
            v = self.f(x, yp, F_value)            # v_n is a (N_data, N) matrix 
            V = pt.cat((V, v[:,None,:]), dim=1)   # v_n size (N_data, n, N)

        yp = self._N(y, V, self.inner_iterations)
        return x + yp # y = x_{k+1} - x_k
    
    def _N(self, y, V, n):
        lower_index = ( (n-1) * n ) // 2
        upper_index = ( n * (n+1) ) // 2
        return y + pt.tensordot(V, self.weights[lower_index:upper_index], dims=([1],[0]))
    
class NewtonKrylovNetwork(nn.Module):
    def __init__(self, F, inner_iterations):
        super(NewtonKrylovNetwork, self).__init__()

        self.F = F
        self.inner_layer = NewtonKrylovLayer(F, inner_iterations)
        
        # This network is just one inner layer
        layer_list = [('layer_0', self.inner_layer)]
        self.layers = pt.nn.Sequential(OrderedDict(layer_list))

        # Check the number of parameters
        print('Number of Newton-Krylov Parameters:', sum(p.numel() for p in self.parameters()))

    # Input x must have size (N_data, N)
    def forward(self, x):
        return self.layers(x)
    
class NewtonKrylovLoss(nn.Module):
    def __init__(self, network, F, outer_iterations, base_weight=4.0):
        super(NewtonKrylovLoss, self).__init__()
        
        self.network = network
        self.F = F
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