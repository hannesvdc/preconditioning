import numpy as np
import numpy.linalg as lg

from api.Scheduler import ConstantScheduler

class BFGSOptimizer:
    def __init__(self, loss_fn, d_loss_fn, scheduler=ConstantScheduler(0.01)):
        self.f = loss_fn
        self.df = d_loss_fn
        self.scheduler = scheduler
        self.stability_threshold = 1.e-10

        # Keep history of training
        self.losses = []
        self.gradient_norms = []

    def optimize(self, x0, tolerance=None, maxiter=100):
        x = np.copy(x0)
        l = self.f(x)
        g = self.df(x)
        self.losses.append(l)
        self.gradient_norms.append(lg.norm(g))
        print('x =', x, g)

        n_iterations = 0
        I = np.eye(x.size)
        H = np.copy(I)
        while (tolerance is not None and lg.norm(g) >= tolerance) or (tolerance is None and n_iterations < maxiter):
            print('\nEpoch #', n_iterations)
            alpha = self.scheduler.getLearningRate(n_iterations)

            # Descent stepping
            pk = -np.dot(H, g)
            sk = alpha * pk
            print('pk', pk)
            print('sk', sk)

            # Updating position and gradient
            xp = x + sk
            gp = self.df(xp)

            # BFGS update
            yk = gp - g
            rhok_inv = np.dot(sk, yk)
            if np.abs(rhok_inv) < self.stability_threshold:
                print('Precision Loss in BFGS update. Asssuming rhok is large. Rhok_inv =', rhok_inv)
                rhok = 1000.0
            else:
                rhok = 1.0 / rhok_inv
            A1 = I - np.outer(sk, yk) * rhok
            A2 = I - np.outer(yk, sk) * rhok
            H = np.dot(A1, np.dot(H, A2)) + rhok * np.outer(sk, sk)
            
            # Keeping track of variables for next iteration
            print('new x', xp, x, sk)
            x = np.copy(xp)
            l = self.f(x)
            g = np.copy(gp)

            self.losses.append(l)
            self.gradient_norms.append(lg.norm(g))
            n_iterations += 1
            print('Loss =', l)
            print('Gradient Norm =', lg.norm(g))
            print('Weights =', x)

        print('\nBFGS Optimzer Converged in', n_iterations, 'Epochs! Final Loss =', l)
        return x