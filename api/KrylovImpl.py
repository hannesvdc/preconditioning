import torch as pt
import torch.nn as nn

from collections import OrderedDict

class KrylovLayer(nn.Module):
    def __init__(self, A, inner_iterations):
        super(KrylovLayer, self).__init__()

        self.A = A
        self.f = lambda x, b: pt.matmul(A, x.transpose(0,1)).transpose(0,1) - b

        self.inner_iterations = inner_iterations
        self.n_weights = (inner_iterations * (inner_iterations + 1) ) // 2
        weights = pt.zeros(self.n_weights)
        self.weights = nn.Parameter(weights)

    def forward(self, input):
        x = input[0]
        b = input[1]

        V = self.f(x, b)[:,None,:]
        for n in range(1, self.inner_iterations):
            xp = self._N(x, V, n)
            v = self.f(xp, b)
            V = pt.cat((V, v[:,None,:]), dim=1)

        return self._N(x, V, self.inner_iterations)
    
    def _N(self, y, V, n):
        lower_index = ( (n-1) * n ) // 2
        upper_index = ( n * (n+1) ) // 2
        return y + pt.tensordot(V, self.weights[lower_index:upper_index], dims=([1],[0]))
    
class KrylovNetwork(nn.Module):
    def __init__(self, A, inner_iterations):
        super(KrylovNetwork, self).__init__()
        
        # This network is just one inner layer
        self.inner_layer = KrylovLayer(A, inner_iterations)
        layer_list = [('layer_0', self.inner_layer)]
        self.layers = pt.nn.Sequential(OrderedDict(layer_list))

        # Check the number of parameters
        print('Number of Krylov Parameters:', sum(p.numel() for p in self.parameters()))

    def forward(self, input):
        return self.layers(input)
    
class KrylovLoss(nn.Module):
    def __init__(self, network, outer_iterations, base_weight=4.0):
        super(KrylovLoss, self).__init__()
        
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