import autograd.numpy as np
import autograd.numpy.linalg as lg
import scipy.optimize as opt

def testInnerArguments():
    f = lambda x: np.exp(x**2) - 1.0
    x0 = 3.0
    
    class GMRES_Counter:
        def __init__(self):
            self.counter = 0
    instance_counter = GMRES_Counter()

    def _inner_callback(gmres_counter, args):
        print('here')
        gmres_counter.counter += 1
    inner_cb = lambda x: _inner_callback(instance_counter, x)

    arguments = {'inner_callback': inner_cb, 'inner_callback_type': 'x'}
    solution = opt.newton_krylov(f, x0, verbose=True, method='gmres', inner_callback=inner_cb, inner_callback_type='x')
    print('Solution', solution)
    print('Inner gmres iterations',instance_counter.counter)

if __name__ == '__main__':
    testInnerArguments()