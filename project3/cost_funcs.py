# list over all cost functions to save space in the NN file
import numpy as np
import autograd.numpy as np

def grad_ols(y, a, **kwargs):
    return (a-y)/y.shape[0]

def grad_ridge(y, a,w, **kwargs):
    return (a-y)/y.shape[0] + kwargs['lambda']*np.sum(w,axis=0).T

def grad_lasso(y,a,w,**kwargs):
    return (a-y)/y.shape[0] + kwargs['lambda']*np.sum(np.sign(w, axis=0)).T

def grad_cross_entropy(y, a, **kwargs):
    return (a - y)/(y.shape[0]*a*(1 - a) )

def grad_cross_entropy_l2reg(y, a, w, **kwargs):
    return (a - y)/( y.shape[0]*a*(1 - a) ) + kwargs['lambda']*np.sum(w,axis=0).T

def grad_cross_entropy_l1reg(y, a, w, **kwargs):
    return (a - y)/( y.shape[0]*a*(1 - a) ) + kwargs['lambda']*np.sum(np.sign(w, axis=0)).T

def cost_ols(y, X, theta, **kwargs):
    return .5*np.sum(np.square(y-X @ theta))/y.shape[0]

def cost_ridge(y, X, theta, **kwargs):
    lmbda = kwargs['lambda']
    return .5*np.sum(np.square(y-X @ theta))/y.shape[0] + lmbda*np.sum(np.square(theta))

def cost_lasso(y, X, theta, **kwargs):
    lmbda = kwargs['lambda']
    return .5*np.sum(np.square(y-X @ theta))/y.shape[0]+ lmbda*np.sum(np.abs(theta))

def cross_entropy(y, a, **kwargs):
    # add epsilon in log to avoid divide by zero
    epsilon = 1e-10
    return - np.sum( y*np.log(a + epsilon) + (1-y)*np.log(1-a+epsilon) )/y.shape[0]

def cross_entropy_l2reg(y, a, w, **kwargs):
    lmbda = kwargs['lambda']
    # add epsilon in log to avoid divide by zero
    epsilon = 1e-10
    return - np.sum( y*np.log(a+epsilon) + (1-y)*np.log(1-a+epsilon) )/y.shape[0] + .5* lmbda*np.sum(np.square(w))

def cross_entropy_l1reg(y, a, w, **kwargs):
    lmbda = kwargs['lambda']
    # add epsilon in log to avoid divide by zero
    epsilon = 1e-10
    return - np.sum( y*np.log(a+epsilon) + (1-y)*np.log(1-a+epsilon) )/y.shape[0] + lmbda*np.sum(np.abs(w))
