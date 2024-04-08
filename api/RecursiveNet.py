import autograd.numpy as np
import autograd.numpy.linalg as lg

class R2N2:
    def __init__(self, A_data, b_data, outer_iterations, inner_iterations, baseweight=4.0):
        self.f = lambda x, A, b: np.dot(A, x) - b
        self.M = A_data.shape[0]

        self.A_data = A_data # Matrix with A_i in the first two dimensions
        self.b_data = b_data # Matrix with b_i as columns

        self.outer_iterations = outer_iterations
        self.inner_iterations = inner_iterations
        self.baseweight = baseweight

    def loss(self, weights): # Weights is a vector with n_inner * (n_inner+1) // 2 elements
        total_loss = 0.0
        N = self.b_data.shape[1]

        for n in range(N):
            x = np.zeros(self.M)
            A = self.A_data[:,:,n]
            b = self.b_data[:,n]
            
            for k in range(1, self.outer_iterations+1): # do self.outer_iterations iterations
                loss_weight = self.baseweight**k
                x = self.inner_forward(x, A, b, weights)
                total_loss += loss_weight * lg.norm(self.f(x, A, b))**2

        averaged_loss = total_loss / N
        return averaged_loss
    
    def forward(self, weights, A, b, n_outer_iterations=None):
        x = np.zeros(self.M)
        if n_outer_iterations is None:
            n_outer_iterations = self.outer_iterations
        
        samples = [x]
        for k in range(1, n_outer_iterations+1):
            x = self.inner_forward(x, A, b, weights)
            samples.append(x)

        return samples

    def inner_forward(self, x, A, b, weights):
        V = np.array([self.f(x, A, b)]).transpose() # v_0

        for n in range(1, self.inner_iterations): # do inner_iterations-1 function evaluations
            xp = self._N(x, V, n, weights)
            v = self.f(xp, A, b) # v_n
            V = np.append(V, np.array([v]).transpose(), axis=1)

        return self._N(x, V, self.inner_iterations, weights)
    
    def _N(self, x, V, n, weights):
        lower_index = ( (n-1) * n ) // 2
        upper_index = ( n * (n+1) ) // 2
        return x + np.dot(V, weights[lower_index:upper_index])
    


    
