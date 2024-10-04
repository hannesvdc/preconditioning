import torch as pt
import torch.nn as nn

from collections import OrderedDict

class NormalisedKrylovLayer(nn.Module):
    def __init__(self, A, inner_iterations):
        super(NormalisedKrylovLayer, self).__init__()

        self.A = A.T
        self.f = lambda x, b: b - pt.matmul(x, self.A)

        self.inner_iterations = inner_iterations
        self.n_weights = (inner_iterations * (inner_iterations + 1) ) // 2
        weights = pt.normal(pt.zeros(self.n_weights), 0.1*pt.ones(self.n_weights))
        #weights = pt.zeros(self.n_weights)
        self.weights = nn.Parameter(weights)

    def forward(self, input):
        x0 = input[0]
        b = input[1]

        # Storage for Krylov vectors
        V = pt.empty([x0.shape[0], 0, x0.shape[1]])

        r0 = self.f(x0, b)
        r0_norm = pt.norm(r0, dim=1, keepdim=True)
        v = self._N(V, r0, 0)
        #v = v / pt.norm(v, dim=1, keepdim=True)
        V = pt.cat((V, v[:,None,:]), dim=1)

        for n in range(1, self.inner_iterations):
            x_in = x0 + r0_norm * v # v = V[:,-1,:]
            r_n = self.f(x_in, b)

            v = self._N(V, r_n, n)
            #v = v / pt.norm(v, dim=1, keepdim=True)
            V = pt.cat((V, v[:,None,:]), dim=1)

        return x0 + r0_norm * self._N_final(V, self.inner_iterations)
    
    def _N(self, V, r, n):
        lower_index = ( (n-1) * n ) // 2
        upper_index = ( n * (n+1) ) // 2
        return pt.tensordot(V, self.weights[lower_index:upper_index], dims=([1],[0])) + r / pt.norm(r, dim=1, keepdim=True)
    
    def _N_final(self, V, n):
        lower_index = ( (n-1) * n ) // 2
        upper_index = ( n * (n+1) ) // 2
        return pt.tensordot(V, self.weights[lower_index:upper_index], dims=([1],[0]))

class NormalisedKrylovNetwork(nn.Module):
    def __init__(self, A, inner_iterations):
        super(NormalisedKrylovNetwork, self).__init__()
        
        # This network is just one inner layer
        self.inner_layer = NormalisedKrylovLayer(A, inner_iterations)
        layer_list = [('layer_0', self.inner_layer)]
        self.layers = pt.nn.Sequential(OrderedDict(layer_list))

        # Check the number of parameters
        print('Number of Krylov Parameters:', sum(p.numel() for p in self.parameters()))

    def forward(self, input):
        return self.layers(input)
    
class NormalisedKrylovLoss(nn.Module):
    def __init__(self, network, outer_iterations, base_weight=4.0):
        super(NormalisedKrylovLoss, self).__init__()
        
        self.network = network
        self.f = network.inner_layer.f
        self.outer_iterations = outer_iterations
        self.base_weight = base_weight

    def forward(self, b):
        x = pt.zeros_like(b)

        loss = 0.0
        N_data = x.shape[0]
        for k in range(self.outer_iterations):
            input = (x, b)
            x = self.network.forward(input)

            loss_weight = self.base_weight**(k+1)
            loss += loss_weight * pt.sum(pt.square(self.f(x, b)))

        avg_loss = loss / N_data
        return avg_loss