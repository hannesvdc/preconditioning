import torch as pt
import torch.nn as nn

class InverseJacobianLayer(nn.Module):
    """ Custom Layer that computes J^{-1} rhs using a Krylov Neural-Network """
    def __init__(self, F_macro, 
                       inner_iterations : int):
        super(InverseJacobianLayer, self).__init__()

        # Computes directional derivative via normed vectors
        self.eps = 1.e-8
        self.F = F_macro # the function we want to solve, needs to n
        self.dF_v = lambda xk, w, Fxk: (self.F(xk + self.eps*w) - Fxk) / self.eps
        self.f = lambda xk, w, rhs, Fxk: self.dF_v(xk, w, Fxk) - rhs # The linear system during every outer iteration

        self.inner_iterations = inner_iterations
        self.n_weights = (inner_iterations * (inner_iterations + 1) ) // 2
        weights = pt.zeros(self.n_weights)
        self.weights = nn.Parameter(weights)

    # Input = (xk, rhs) or (xk, rhs, w)
    def forward(self, input):
        xk = input[0]
        rhs = input[1]
        Fxk = self.F(xk)
        if len(input) > 2:
            w = input[2]
        else:
            w = pt.zeros_like(xk)

        V = self.f(xk, w, rhs, Fxk)[:,None,:]
        for n in range(1, self.inner_iterations):
            wp = self._N(w, V, n)
            v = self.f(xk, wp, rhs, Fxk)
            V = pt.cat((V, v[:,None,:]), dim=1)

        wp = self._N(w, V, self.inner_iterations)
        return wp
    
    def _N(self, w, V, n):
        lower_index = ( (n-1) * n ) // 2
        upper_index = ( n * (n+1) ) // 2
        return w + pt.tensordot(V, self.weights[lower_index:upper_index], dims=([1],[0]))
    
class PreconditionedNewtonKrylovLayer(nn.Module):
    """ Custom Preconditioned Newton-Krylov layer to solve M * (JF(xk) y = -F(xk)) """
    def __init__(self, F, 
                       inner_iterations : int, 
                       inv_jac_layer : InverseJacobianLayer):
        super(PreconditionedNewtonKrylovLayer, self).__init__()
        self.F = F

        self.eps = 1.e-8
        self.dF_v = lambda x, v, Fx: (self.F(x + self.eps*v) - Fx) / self.eps
        self.f = lambda x, v, Fx: self.dF_v(x, v, Fx) + Fx
        
        self.inner_iterations = inner_iterations
        self.n_weights = (inner_iterations * (inner_iterations + 1) ) // 2
        weights = pt.zeros(self.n_weights)
        self.weights = nn.Parameter(weights)

        self.inv_jac_layer = inv_jac_layer

    def forward(self, xk):
        F_value = self.F(xk) 
        self.inv_jac_layer.computeFValue(xk)

        y = pt.zeros_like(xk)
        V = pt.empty((xk.shape[0], 0, xk.shape[1]))
        for n in range(self.inner_iterations):
            yp = self._N(y, V, n)
            v = self.f(xk, yp, F_value)
            w = self.inv_jac_layer((xk, v))
            V = pt.cat((V, w[:,None,:]), dim=1)

        yp = self._N(y, V, self.inner_iterations)
        return xk + yp
    
    def _N(self, y, V, n):
        lower_index = ( (n-1) * n ) // 2
        upper_index = ( n * (n+1) ) // 2
        return y + pt.tensordot(V, self.weights[lower_index:upper_index], dims=([1],[0]))

class PreconditionedNewtonKrylovNetwork(nn.Module):
    def __init__(self, F, 
                       F_macro, 
                       inner_iterations : tuple[int, int]):
        super(PreconditionedNewtonKrylovNetwork, self).__init__()
        
        self.inv_jac_layer = InverseJacobianLayer(F_macro, inner_iterations[1])
        self.inner_layer = PreconditionedNewtonKrylovLayer(F, inner_iterations[0], self.inv_jac_layer)

        self.params = []
        self.params.extend(self.inv_jac_layer.parameters())
        self.params.extend(self.inner_layer.parameters())

        print('Total Number of Preconditioned Newton-Krylov Parameters:', sum(p.numel() for p in self.parameters()))

    def forward(self, xk):
        return self.inner_layer(xk)

class PreconditionedNewtonKrylovLoss(nn.Module):
    def __init__(self, network : PreconditionedNewtonKrylovNetwork, 
                       F, 
                       outer_iterations : int, 
                       base_weight : float=4.0):
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
            loss += loss_weight * pt.sum(pt.square(self.F(x)))

        avg_loss = loss / N_data
        return avg_loss