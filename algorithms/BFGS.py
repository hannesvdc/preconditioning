import numpy as np
import numpy.linalg as lg

from api.Scheduler import ConstantScheduler

class BFGSOptimizer:
    def __init__(self, loss_fn, d_loss_fn, scheduler=ConstantScheduler(0.01)):
        self.f = loss_fn
        self.df = d_loss_fn
        self.scheduler = scheduler

        # Keep history of training
        self.losses = []
        self.gradient_norms = []

    def optimize(self, x0, n_epochs=100):
        x = np.copy(x0)
        l = self.f(x)
        g = self.df(x)
        self.losses.append(l)
        self.gradient_norms.append(lg.norm(g))

        n_iterations = 0
        H = np.eye(len(x))
        while n_iterations < n_epochs:
            print('\nEpoch #', n_iterations)
            alpha = self.scheduler.getLearningRate(n_iterations)

            # Descent stepping
            p = -np.dot(H, g)
            xp = x + alpha * p
            gp = self.df(x)

            # BFGS update
            s = xp - x
            y = gp - g
            denom = np.dot(s, y)
            I = np.eye(len(s))
            H = np.matmul(I - np.outer(s, y) / denom, np.matmul(H, I - np.outer(y, s) / denom)) + np.outer(s, s) / denoms
            
            # Keeping track of variables for next iteration
            x = np.copy(xp)
            l = self.f(x)
            g = np.copy(gp)

            self.losses.append(l)
            self.gradient_norms.append(lg.norm(g))
            n_iterations += 1
            print('Loss =', l)
            print('Gradient Norm =', lg.norm(g))
            print('Weights =', x)

        print('\BFGS Optimzer Converged in', n_iterations, 'Epochs! Final Loss =', l)
        return x