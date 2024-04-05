import autograd.numpy as np
import autograd.numpy.linalg as lg

class R2N2:
    def __init__(self, A, outer_iterations, inner_iterations, data):
        self.A = A
        self.f = lambda x, b: np.dot(A, x) - b
        self.M = self.A.shape[0]

        self.outer_iterations = outer_iterations
        self.inner_iterations = inner_iterations
        self.data = data # Matrix with b_i as columns

    def loss(self, weights): # Weights is a vector with n_inner * (n_inner+1) // 2 elements
        total_loss = 0.0
        N = self.data.shape[1]

        for n in range(N):
            x = np.zeros(self.M)
            b = self.data[:,n]
            
            for k in range(1, self.outer_iterations+1):
                loss_weight = 4.0**k
                x = self.inner_forward(x, b, weights)
                total_loss += loss_weight * lg.norm(self.f(x, b))**2

        averaged_loss = total_loss / N
        return averaged_loss
    
    def forward(self, weights, b, n_outer_iterations=None):
        x = np.zeros_like(b)
        if n_outer_iterations is None:
            n_outer_iterations = self.outer_iterations
        
        samples = [x]
        for k in range(1, n_outer_iterations+1):
            x = self.inner_forward(x, b, weights)
            samples.append(x)

        return samples

    def inner_forward(self, x, b, weights):
        V = np.array([-b]).transpose() # This is technically the first function evaluations, but we do not count it.

        for n in range(1, self.inner_iterations): # do inner_iterations-1 function evaluations
            xp = self._N(x, V, n, weights)
            v = self.f(xp, b)
            V = np.append(V, np.array([v]).transpose(), axis=1)

        return self._N(x, V, self.inner_iterations, weights) # Does this need to be xp? Think!
    
    def _N(self, x, V, n, weights):
        lower_index = ( (n-1) * n ) // 2
        upper_index = ( n * (n+1) ) // 2
        return x + np.dot(V, weights[lower_index:upper_index])
    


    
