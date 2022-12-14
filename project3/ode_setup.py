# Script where the cost functions for the different ODEs are specified
from autograd import grad, elementwise_grad
import numpy as np
import autograd.numpy as np
def trial_func(X, P, neural_network, init_cond):
    neural_network.bias = P[0]
    neural_network.weights = P[1]
    pred = neural_network.predict(X)
    print("pred : ", pred)
    print("stop printing from inside the trial function")
    return init_cond + X * pred # the training prediction

def ode_rhs(gtrial, alpha):
    return alpha * gtrial

def cost_func_ode(X, P, neural_network, init_cond, alpha):
    # evaluation of the deriv wrt x trial function
    lhs = elementwise_grad(trial_func, 0)(X, P, neural_network, init_cond)
    # eval trial function
    gtrial = trial_func(X, P, neural_network, init_cond) 
    
    # evaluation of the rhs of the ODE
    rhs = ode_rhs(gtrial, alpha)
    
    # computing the cost
    cost = np.sum((lhs - rhs)**2)
    return cost/np.size(lhs)

def analytic_sol(t, g0, alpha):
    return g0 * np.exp(alpha*t)
