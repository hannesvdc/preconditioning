import numpy as np
import numpy.linalg as lg

from api.Scheduler import ConstantScheduler

class AdamOptimizer:
    def __init__(self, loss_fn, d_loss_fn, scheduler=ConstantScheduler(0.001), beta1=0.9, beta2=0.99, epsilon=1.e-8):
        self.f = loss_fn
        self.df = d_loss_fn

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scheduler = scheduler

        # Keep history of training
        self.losses = []
        self.gradient_norms = []

    def optimize(self, x0, n_epochs=100, tolerance=1.e-8):
        x = np.copy(x0)
        l = self.f(x)
        g = self.df(x)
        self.losses.append(l)
        self.gradient_norms.append(lg.norm(g))

        m = 0.0
        v = 0.0
        n_iterations = 0
        while n_iterations < n_epochs and lg.norm(g) > tolerance:
            print('\nEpoch #', n_iterations)
            alpha = self.scheduler.getLearningRate(n_iterations)

            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * np.dot(g, g)

            mp = m / (1.0 - self.beta1**(n_iterations + 1))
            mv = v / (1.0 - self.beta2**(n_iterations + 1))

            x = x - alpha * mp / (np.sqrt(mv) + self.epsilon)
            l = self.f(x)
            g = self.df(x)

            self.losses.append(l)
            self.gradient_norms.append(lg.norm(g))
            n_iterations += 1
            print('Loss =', l)
            print('Gradient Norm =', lg.norm(g))

        print('\nAdam Optimzer Converged in', n_iterations, 'Epochs! Final Loss =', l)
        return x