import autograd.numpy as np

import internal.SuperStructure

class NewtonKrylovSuperStructure(internal.SuperStructure):
    def __init__(self, f, M, data, outer_iterations, inner_iterations, baseweight=4.0):
        super().__init__()

        self.f = f
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
    
    def forward(self, weights, d, n_outer_iterations=None):
        x = np.zeros(self.M)
        if n_outer_iterations is None:
            n_outer_iterations = self.outer_iterations
        
        samples = [x]
        for _ in range(1, n_outer_iterations+1):
            x = self.inner_forward(x, d, weights)
            samples.append(x)

        return samples
    
    def inner_forward(self, x, d, weights):
        pass