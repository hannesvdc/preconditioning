import numpy as np
import numpy.linalg as lg

from api.Scheduler import ConstantScheduler

class GradientDescentOptimizer:
    def __init__(self, loss_fn, d_loss_fn, scheduler=ConstantScheduler(0.001)):
        self.f = loss_fn
        self.df = d_loss_fn

        self.scheduler = scheduler

        # Keep history of training
        self.losses = []
        self.gradient_norms = []
        self.lastweights = None

    def getName(self):
        return 'Gradient Descent'

    def optimize(self, x0, n_epochs=100, tolerance=1.e-8):
        x = np.copy(x0)
        l = self.f(x)
        g = self.df(x)
        norm_g = lg.norm(g)

        self.losses.append(l)
        self.gradient_norms.append(norm_g)
        self.lastweights = x

        for n_iterations in range(n_epochs):
            print('\nEpoch #', n_iterations)
            alpha = self.scheduler.getLearningRate(n_iterations)
            x = x - alpha * g

            l = self.f(x)
            g = self.df(x)
            norm_g = lg.norm(g)

            if norm_g < tolerance:
                print('\nAdam Optimzer Converged in', n_iterations, 'Epochs! Final Loss =', l)
                return x

            self.losses.append(l)
            self.gradient_norms.append(norm_g)
            self.lastweights = x
            print('Loss =', l)
            print('Gradient Norm =', norm_g)
            print('Weights =', x)

        print('\Gradient Descent Reached Maximum Number of Epochs ', n_epochs, '. Final Loss =', l)
        return x