import torch as pt
import torch.nn as nn

from collections import OrderedDict
    
class PreconditionedNewtonKrylovLayer(nn.Module):
    """ Custom Preconditioned Newton-Krylov layer to solve M * (JF(xk) y = -F(xk)). 

        M_generator(xk, F_value) is a function that takes a (N_data, data_size) tensor and 
        returns an (N_data, data_size, data_size) tensor with the Jacobian of a 
        macroscopic function F_macro(xk) in the second and third dimensions. 
    """
    def __init__(self, F, inner_iterations, M_generator):
        super(PreconditionedNewtonKrylovLayer, self).__init__()
        self.F = F
        print(F, inner_iterations, M_generator)

        self.eps = 1.e-8
        #self.dF_v = lambda x, v, Fx: (self.F(x + self.eps*v) - Fx) / self.eps
        self.f = lambda x, v, Fx: self.dF_v(x, v, Fx) + Fx
        
        self.inner_iterations = inner_iterations
        self.n_weights = (inner_iterations * (inner_iterations + 1) ) // 2
        weights = pt.zeros(self.n_weights)
        self.weights = nn.Parameter(weights)

        self.M_generator = M_generator

    def dF_v(self, x, v, Fx):
        a = self.F(x + self.eps*v)
        print('input = ', x + self.eps*v)
        return (a - Fx) / self.eps
        

    """ xk has shape (N_data, data_size) """
    def forward(self, xk):
        print('xk', xk)
        F_value = self.F(xk) # Shape (N_data, data_size)
        print('F_value', F_value)
        (LU, pivots) = pt.linalg.lu_factor(self.M_generator(xk, F_value), pivot=True) # LU has shape (N_data, data_size, data_size)

        y = pt.zeros_like(xk)
        V = pt.empty((xk.shape[0], 0, xk.shape[1])) # Shape (N_data, 0, data_size)
        for n in range(self.inner_iterations):
            yp = self._N(y, V, n)
            print('yp', yp)
            v = self.f(xk, yp, F_value) # Shape (N_data, data_size)
            print('v', v)
            w = pt.linalg.lu_solve(LU, pivots, v[:,:,None])[:,:,0]
            print('w', w)
            V = pt.cat((V, w[:,None,:]), dim=1) # Shape (N_data, n, data_size)

        yp = self._N(y, V, self.inner_iterations)
        print('yp', yp)
        return xk + yp
    
    def _N(self, y, V, n):
        lower_index = ( (n-1) * n ) // 2
        upper_index = ( n * (n+1) ) // 2
        return y + pt.tensordot(V, self.weights[lower_index:upper_index], dims=([1],[0]))

class PreconditionedNewtonKrylovNetwork(nn.Module):
    def __init__(self, F, 
                       inner_iterations,
                       M_generator):
        super(PreconditionedNewtonKrylovNetwork, self).__init__()
        
        self.inner_layer = PreconditionedNewtonKrylovLayer(F, inner_iterations, M_generator)
        layer_list = [('layer_0', self.inner_layer)]
        self.layers = pt.nn.Sequential(OrderedDict(layer_list))

        print('Total Number of Preconditioned Newton-Krylov Parameters:', sum(p.numel() for p in self.parameters()))

    def forward(self, xk):
        return self.inner_layer(xk)

class PreconditionedNewtonKrylovLoss(nn.Module):
    def __init__(self, network,
                       outer_iterations,
                       base_weight=4.0):
        super(PreconditionedNewtonKrylovLoss, self).__init__()
        
        self.network = network
        self.F = network.inner_layer.F
        self.outer_iterations = outer_iterations
        self.base_weight = base_weight

    def forward(self, x):
        loss = 0.0
        N_data = x.shape[0]

        for k in range(self.outer_iterations):
            x = self.network.forward(x)

            loss_weight = self.base_weight**(k+1)
            loss += loss_weight * pt.sum(pt.square(self.F(x)))

        avg_loss = loss / N_data
        return avg_loss