# Script where the cost functions for the different ODEs are specified
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
import numpy as np
def trial_func(init_cond, X, P, neural_network):
    neural_network.Xdata_full = X # set design matrix
    neural_network.bias = P[0]       # in dict structure { layer : bias vector (N_l, 1) } 
    neural_network.weights = P[1]    # in dict structure { layer : weight matrix (N_{l-1} , N_l) } 
    return init_cond + neural_network.predict(X) # the training prediction

def deriv_trial_func(gtrail, alpha):
    return alpha * gtrial

def cost_func(t, params,neural_network, init_cond):
    # evaluation of the deriv trial function
    lhs = elementwise_grad(trial_func, 0)(t, P)
    
    # eval trial function
    gtrial = trial_func(init_cond, t, params, neural_network) 
    
    # deriv wrt argument of original function
    rhs = deriv_trial_func(gtrial)
    
    # computing the cost
    cost = np.sum((lhs - rhs)**2)/np.size(lhs)
    return cost

def analytic_sol(t, g0, alpha):
    return g0 * np.exp(alpha*t)
