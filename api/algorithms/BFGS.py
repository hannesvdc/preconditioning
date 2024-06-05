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

    def optimize(self, x0, tolerance=None, maxiter=1000):
        x = np.copy(x0)
        l = self.f(x)
        g = self.df(x)
        norm_g = lg.norm(g)

        self.losses.append(l)
        self.gradient_norms.append(norm_g)

        I = np.eye(x.size)
        H = np.copy(I)
        for n_iterations in range(maxiter):
            print('\nEpoch #', n_iterations)
            alpha = self.scheduler.getLearningRate(n_iterations)

            # Descent stepping
            pk = -np.dot(H, g)
            sk = alpha * pk

            # Updating position and gradient
            xp = x + sk
            lp = self.f(xp)
            gp = self.df(xp)
            norm_gp = lg.norm(gp)
            if not np.isfinite(lp):
                print('Invalid Value Encountered. Aborting')
                return xp
            if tolerance is not None and norm_gp < tolerance:
                print('\nBFGS Optimzer Converged in', n_iterations, 'Epochs! Final Loss =', lp)
                return xp

            # BFGS update
            yk = gp - g
            rhok_inv = np.dot(sk, yk)
            if rhok_inv == 0.0:
                print('Precision Loss in BFGS update. Asssuming rhok is large. Rhok_inv =', rhok_inv)
                rhok = 1000.0
            else:
                rhok = 1.0 / rhok_inv
            A1 = I - np.outer(sk, yk) * rhok
            A2 = I - np.outer(yk, sk) * rhok
            H = np.dot(A1, np.dot(H, A2)) + rhok * np.outer(sk, sk)
            
            # Keeping track of variables for next iteration
            x = np.copy(xp)
            l = lp
            g = np.copy(gp)
            norm_g = norm_gp

            # Keep internal history. Should we pre-allocate losses and gradient_norms for efficiency?
            self.losses.append(l)
            self.gradient_norms.append(norm_g)
            print('Loss =', l)
            print('Gradient Norm =', norm_g)
            print('Weights =', x)

        if tolerance is None:
            print('\nBFGS Optimzer Converged in', n_iterations, 'Epochs! Final Loss =', l)
        else:
            print('Maximum Number of Iterations Reached. Returning Current Iterate.')
        return x