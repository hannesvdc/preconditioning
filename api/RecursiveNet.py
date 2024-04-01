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

    def loss(self, weights):
        averaged_loss = 0.0
        N = self.data.shape[1]

        for n in range(N):
            x = np.zeros(self.M)
            b = self.data[:,]
            
            for k in range(1, self.outer_iterations+1):
                loss_weight = 4.0**k
                x = self.inner_forward(x, b, weights)
                averaged_loss += loss_weight * lg.norm(self.f(x, b))**2

        averaged_loss = averaged_loss / N
        return averaged_loss

    def inner_forward(self, x, b, weights):
        V = np.zeros((self.M, self.inner_iterations))
        V[:,0] = -b

        for n in range(1, self.inner_iterations):
            xp = self._N(x, V, n, weights)
            v = self.f(xp, b)
            V[:,n] = v

        return self._N(x, V, self.inner_iterations, weights)
    
    def _N(self, x, V, n, weights):
        return x + np.dot(V, weights[n,:])
    


    
