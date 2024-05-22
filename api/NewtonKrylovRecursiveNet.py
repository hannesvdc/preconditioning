import autograd.numpy as np

import internal.SuperStructure

class NewtonKrylovSuperStructure(internal.SuperStructure):
    def __init__(self, f, M, data, outer_iterations, inner_iterations, baseweight=4.0):
        super().__init__()

        self.eps = 1.e-8

        self.f = f
        self.dfv = lambda x, v: (self.f(x + self.eps*v) - self.f(x)) / self.eps
        self.F = lambda x, v: self.dfv(x, v) - self.f(x)
        self.M = M

        self.data = data
        self.N_data = data.shape[1]

        self.outer_iterations = outer_iterations
        self.inner_iterations = inner_iterations
        self.baseweight = baseweight

    def loss(self, weights):
        total_loss = 0.0

        for n in range(self.N_data):
            d = self.data[:,n]
            x = np.zeros(self.M)

            for k in range(self.outer_iterations): # do self.outer_iterations iterations
                loss_weight = self.baseweight**(k+1)
                x = self.inner_forward(x, d, weights)
                total_loss += loss_weight * self.stable_normsq(self.f_loss(x, d))

        averaged_loss = total_loss / self.N_data
        return averaged_loss
    
    def forward(self, weights, n_outer_iterations=None):
        x = np.zeros(self.M) # x = x_k = solutition to self.f(x) = -0
        if n_outer_iterations is None:
            n_outer_iterations = self.outer_iterations
        
        samples = [x]
        for _ in range(1, n_outer_iterations+1):
            y = self.inner_forward(x, weights) # Solution to linear systems
            x = x + y # y = x_{k+1} - x_k
            samples.append(x)

        return samples
    
    # One complete inner iterations
    def inner_forward(self, xk, weights):
        y = np.zeros_like(xk) # y stores the variable that solves F(xk, y) = 0 (i.e. the linear system)
        V = np.array([self.F(xk, y)]).transpose() # v_0

        for n in range(1, self.inner_iterations): # do inner_iterations-1 function evaluations
            yp = self._N(y, V, n, weights)
            v = self.F(xk, yp) # v_n
            V = np.append(V, np.array([v]).transpose(), axis=1)

        return self._N(y, V, self.inner_iterations, weights)
    
    def _N(self, y, V, n, weights):
        lower_index = ( (n-1) * n ) // 2
        upper_index = ( n * (n+1) ) // 2
        return y + np.dot(V, weights[lower_index:upper_index])