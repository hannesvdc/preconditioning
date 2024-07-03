import autograd.numpy as np

class SuperStructure:
    def __init__(self):
        self.stable_normsq = lambda r: np.dot(r,r)

    def loss(self, weights):
        pass

    def forward(self, **args):
        pass
