# to test the stripped optimizer from project 2
from project3.optimization import *
from project3.cost_funcs import *
import numpy as np
from autograd import grad
import matplotlib.pyplot as plt

x = np.linspace(-1,1,100)
X = np.c_[np.ones(x.shape), x, x**2]
y = (x**2 + x)[:,np.newaxis] # want y as column vector

max_iter = 500
methods = ['grad_desc', 'grad_desc_wmom', 'adagrad', 'rms_prop', 'adam']
for method in methods:
    plt.title(method)
    theta = 2*np.random.rand(X.shape[1])[:,np.newaxis]
    optimizer = optimizers(X, y, cost_ols, eta = 0.01, theta_init = theta, max_iter = 1, w_mom = True)
    count = 1
    for i in range(max_iter):
        optimizer.theta = theta
        optimizer(method = method, X = X, Y = y)
        plt.plot(x, X @ optimizer.theta, alpha = (i+1)/max_iter, c = 'k')
        theta_old = optimizer.theta
    plt.plot(x, y,c = 'r', lw = 3, label = 'OG data')
    plt.plot(x, X @ optimizer.theta,label = 'Last iter', lw = 3, ls = '--', c = 'b')
    plt.legend()
    plt.show()
