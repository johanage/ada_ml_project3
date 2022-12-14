# ODE solver wiht NN implementation from project2
import os
from project3.neural_network import *
from project3.optimization import *
import numpy as np
import autograd.numpy as np
from sklearn.model_selection import train_test_split
from project3.ode_setup import * # import all the functions necessary to setup the ode for a NN
from autograd import grad
import matplotlib.pyplot as plt
# generate the inital data
t = np.linspace(0,5,10)[:,np.newaxis] # s.t. it is easier to observe the exp growth
alpha = 1
g0 = 1
X = np.c_[np.ones(t.shape), t]
Y = np.zeros(t.shape) # the objective is that the difference between rhs and lhs will be zero 
asol = analytic_sol(t, g0, alpha)
#plt.plot(t, asol)
#plt.show()

# hyperparameters of the FFNN
eta = 1e-1
n_hidden_neurons = 10
# default value of size of mini batch for SGD in sklearn is min(200, nsamples)
epochs = 100
batch_size = 5
beta1 = 0.9
beta2 = 0.99
gamma = 0.9
w_mom = True
method = 'grad_desc_wmom'
# init NN and set up hidden and output layer
grad_cost_ode = grad(cost_func_ode, 1)
optimizer = optimizers(X = X, y=Y, method = method, cost_func = cost_func_ode, eta = eta, gamma=gamma, beta1=beta1, beta2=beta2, w_mom = w_mom, max_iter = 1)
nn = Neural_Network(X = X, y = Y, costfunc = 'ode', def_costfunc = cost_func_ode, def_grad_costfunc = grad_cost_ode, 
                    eta=eta, gamma = gamma, beta1 = beta1, beta2 = beta2,
                    w_mom = w_mom, method = method, symbolic_differentiation = True, optimizer = optimizer)
nn.add_layer(nodes = n_hidden_neurons, af = 'sigmoid')
nn.output_layer(af = 'linear')
Ptest = [nn.bias, nn.weights]
#cost_test = cost_func_ode(X=X, P=Ptest, neural_network = nn, init_cond = 1, alpha = 1)
#grad_test = grad_cost_ode(X, Ptest, nn, 1, 1)
#print(grad_test)
nn.SGD(epochs, batch_size, method = method, **{'init_cond' : 1, 'alpha' : 1})


