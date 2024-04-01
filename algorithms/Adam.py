import numpy as np
import numpy.linalg as lg

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.99, epsilon=1.e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.losses = []
        self.gradient_norms = []

    def optimize(self, f, df, x0, n_epochs=100, tolerance=1.e-8):
        x = np.copy(x0)
        l = f(x)
        g = df(x)
        self.losses.append(l)
        self.gradient_norms.append(lg.norm(g))

        m = 0.0
        v = 0.0
        n_iterations = 1
        while n_iterations < n_epochs and lg.norm(g) > tolerance:
            print('\nEpoch #', n_iterations)
            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * np.dot(g, g)

            mp = m / (1.0 - self.beta1**n_iterations)
            mv = v / (1.0 - self.beta2**n_iterations)

            x = x - self.learning_rate * mp / (np.sqrt(mv + self.epsilon))
            l = f(x)
            g = df(x)

            self.losses.append(l)
            self.gradient_norms.append(lg.norm(g))
            n_iterations += 1
            print('Loss =', l)
            print('Gradient Norm =', lg.norm(g))

        print('\nAdam optimzer converged in', n_iterations, 'iterations! Final gradient norm =', lg.norm(g))
        return x