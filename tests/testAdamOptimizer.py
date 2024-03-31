import sys
sys.path.append('../')

import autograd.numpy as np
import internal.Adam as adam

def quadraticTest():
    f  = lambda x: 0.5 * (x - 3.0)**2
    df = lambda x: x - 3.0

    optimizer = adam.AdamOptimizer(learning_rate=0.01)

    x0 = 10.0
    xs = optimizer.optimize(df, x0)

    print('Local Minimum', xs)


if __name__ == '__main__':
    quadraticTest()