import numpy as np

class Scheduler:
    def getLearningRate(self, epoch):
        pass

class ConstantScheduler(Scheduler):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def getLearningRate(self, epoch):
        return self.learning_rate
    
class CosineScheduler(Scheduler):
    def __init__(self, initial_lr, final_lr, T):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.T = T

    def getLearningRate(self, epoch):
        return self.final_lr + 0.5*(self.initial_lr + self.final_lr) * (1.0 + np.cos(epoch * np.pi / self.T))
    
class PiecewiseConstantScheduler(Scheduler):
    def __init__(self, rates):
        self.rates = rates

    def getLearningRate(self, epoch):
        key = max(k for k in self.rates if k <= epoch)
        return self.rates[key]