# ODE solver wiht NN implementation from project2
import os
from project2.neural_network import *
import project2.optimization as opt
import numpy as np
from sklearn.model_selection import train_test_split
from project3.ode_setup import * # import all the functions necessary to setup the ode for a NN
from autograd import grad
import matplotlib.pyplot as plt
# generate the inital data
t = np.linspace(0,5,100)[:,np.newaxis] # s.t. it is easier to observe the exp growth
alpha = 1
g0 = 1
X = np.c_[np.ones(t.shape), t]
Y = analytic_sol(t, g0, alpha)
plt.plot(t, Y)
plt.show()

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
method = 'sgd'
# init NN and set up hidden and output layer
grad_cost = grad(cost_func, 0)
nn = Neural_Network(X,Y, costfunc = 'defined', def_costfunc = cost_func, def_grad_costfunc = grad_cost, 
                    eta=eta, gamma = gamma, beta1 = beta1, beta2 = beta2,
                    w_mom = w_mom, method = method, symbolic_differentiation = True)
nn.add_layer(nodes = n_hidden_neurons, af = 'sigmoid')
nn.output_layer(af = 'linear')



