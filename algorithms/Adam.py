import numpy as np
import numpy.linalg as lg

class AdamOptimizer:
    def __init__(self, loss_fn, d_loss_fn, learning_rate=0.001, beta1=0.9, beta2=0.99, epsilon=1.e-8):
        self.f = loss_fn
        self.df = d_loss_fn

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Keep history of training
        self.losses = []
        self.gradient_norms = []
        self.n_iterations = 0

    def setLearningRate(self, _learning_rate):
        self.learning_rate = _learning_rate

    def optimize(self, x0, n_epochs=100, tolerance=1.e-8):
        start_iterations = self.n_iterations
        x = np.copy(x0)
        l = self.f(x)
        g = self.df(x)
        self.losses.append(l)
        self.gradient_norms.append(lg.norm(g))

        m = 0.0
        v = 0.0
        while self.n_iterations - start_iterations < n_epochs and lg.norm(g) > tolerance:
            print('\nEpoch #', self.n_iterations)
            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * np.dot(g, g)

            mp = m / (1.0 - self.beta1**(self.n_iterations+1))
            mv = v / (1.0 - self.beta2**(self.n_iterations+1))

            x = x - self.learning_rate * mp / (np.sqrt(mv) + self.epsilon)
            l = self.f(x)
            g = self.df(x)

            self.losses.append(l)
            self.gradient_norms.append(lg.norm(g))
            self.n_iterations += 1
            print('Loss =', l)
            print('Gradient Norm =', lg.norm(g))

        print('\nAdam Optimzer Converged in', self.n_iterations, 'Epochs! Final Loss =', l)
        return x