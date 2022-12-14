# a simple neural network
from project3.optimization import *
import matplotlib.pyplot as plt
from project3.optimization import *

import torch
from torch.autograd import grad
from torch.autograd.functional import jacobian, hessian


def activation_funcs(zl, af):
    """
    Args:
    af - string, activation function
    zl - ndarray, neurons weighted input at layer l
    """
    if af == "linear":
        out = zl
    if af == "sigmoid":
        out = 1/(1 + np.exp(-zl))
    if af == "tanh":
        out = np.tanh(zl)
    if af == "relu":
        #out = np.maximum(zl, 1e-8)
        out = np.maximum(zl, np.zeros(zl.shape))
    if af == "leaky_relu":
        out = .5*( (1 - np.sign(zl))*1e-2*zl + (1 + np.sign(zl))*zl)
    if af == "softmax":
        out = np.exp(zl)/np.sum( np.exp(zl), axis=1, keepdims=True)
    return out

import torch.nn as nn
import torch.nn.functional as F


class NN_torch(nn.Module):

    def __init__(self,n_input,n_hidden_neurons):
        """
        For now only logistic AF between hidden layers
        """
        super(NN_torch, self).__init__()
        self.n_hidden_neurons = n_hidden_neurons
        self.nlayers = len(n_hidden_neurons)
        self.n_input = n_input
        self.layer0 = nn.Linear(3,n_hidden_neurons[0])
        for l in range(1,self.nlayers):
            exec("self.layer%i = nn.Linear(n_hidden_neurons[l-1],n_hidden_neurons[l] )"%l)
            
    def forward(self, x):
        x = self.layer0(x)
        #x = self.layer1(x)
        #x = self.layer2(x)
        xd = {}
        for l in range(1,self.nlayers):
            x = torch.sigmoid(x)
            print("layer %i after af"%l, x.shape)
            #exec("print(self.layer%i(x).shape)"%l)
            exec("xd[%i] = self.layer%i(x)"%(l,l) )
            x = xd[l]
        return x

def simple_NN(X,P):# afs):
    """
    X - design matrix, assuming it already includes bias
    y - target data
    P - NN parameters (bias and weights)
    afs - activation functions for each layer
    """
    #nhidden = len(afs)
    nhidden = len(P)-1
    # variable representing the
    # output of the previous layer
    almin1 = X
    for l in range(nhidden):
        wl = P[l]
        zl = almin1 @ wl.T
        al = activation_funcs(zl,'sigmoid')#afs[l])
        almin1 = al
    wL = P[-1]
    zL = almin1 @ wl
    aL = zL
    return aL

def trial_func(x, t, P, L=1):
    """
    assume that x and t are column vectors
    """
    u0x = torch.sin(torch.pi*x/L)
    ut0 = 0
    utL = 0
    X = torch.cat((torch.ones(x.shape), x, t), 1)
    out_nn = torch.mean(simple_NN(X, P), (0), True)
    return (1-t)*u0x + t*x*(L-x)*out_nn


def cost_func_ode(X, P, L=torch.tensor(1.) ):
    # cannot use egrad because the jacobian is not diagonal
    # egrad returns the sum of each column of the jacobian
    # evaluation of the deriv trial function wrt X
    
    trial_hessian  = hessian(trial_func, inputs=(X, P,  L) )
    trial_jacobian = jacobian(trial_func, inputs=(X, P, L) )
    print("shapes hessian, jacobian: ", trial_hessian.shape, trial_jacobian.shape)
    

    # computing the cost
    cost = np.sum((d2u_dx2 - du_dt)**2)/du_dt.shape[0]
    return cost

def solve_ode_neural_network(X, n_hidden_neurons, max_iter, eta, L=torch.tensor(1.)):
    """
    X - design matrix
    y - target data
    P - NN parameters (bias and weights)
    n_hidden_neurons - tuple of # neurons for each layer, lenght of tuple is the # layers
    eta - float, learning rate
    """
    nlayers = len(n_hidden_neurons)
    # init params
    P = [None]*nlayers
    # +1 for adding bias
    P[0] = torch.randn( (n_hidden_neurons[0]+1, 3) )
    for l in range(1, nlayers):
        P[l] = torch.randn( (n_hidden_neurons[l] + 1, n_hidden_neurons[l-1]+1) )
    optimizer_list = [None]*nlayers
    for it in range(max_iter):
        # compute the gradient of the cost function of each layer wrt parameters P
        cost = cost_func_ode(X, tuple(P), L=L)
        nabla_cost = grad(outputs = cost, inputs = (X, tuple(P), L) ) 
        print("l2 norm of cost function", np.sum(cost**2)**.5)
        for l in range(nlayers):
            print("l2 norm of cost function gradient in layer %i"%l, np.sum(nabla_cost[l]**2)**.5)
            if it == 0:
                # init the optimizer for each layer (allowing more complex updates)
                optimizer_list[l] = optimizers(X, None, cost_func_ode, eta, theta_init = P[l])
            optimizer_list[l].ADAM(nabla_cost[l], it+1)
            P[l] = optimizer_list[l].theta
    return P

if __name__ == '__main__':
    np.random.seed(4155)
    # NB! Now the solver only works for same dimension along time and spatial axes
    ## Decide the vales of arguments to the function to solve
    Nt = 50
    Nx = 50
    T = torch.tensor(1.)
    L = torch.tensor(1.)
    t = torch.linspace(0, T, Nt).reshape(-1,1)
    x = torch.linspace(0, L, Nx).reshape(-1,1)
    X = torch.cat((torch.ones(t.shape), x, t), 1)
    ## Set up the initial parameters
    num_hidden_neurons = [20, 10, 10, len(t)]
    nlayers = len(num_hidden_neurons)
    num_iter = int(1e3)
    eta = 4e-3
    #afs = ['sigmoid']*nlayers
    P = solve_ode_neural_network(X, num_hidden_neurons, num_iter, eta, L=L)
    # trial func and analytical solution
    g_dnn_ag = trial_func(x, t,P, L=L)
    g_analytical = torch.sin(torch.pi*x/L)*torch.exp(-(torch.pi/L)**2*t)

    # Find the maximum absolute difference between the solutons:
    diff_ag = torch.max(torch.abs(g_dnn_ag - g_analytical))
    print("The max absolute difference between the solutions is: %g"%diff_ag)
    tmesh, xmesh = torch.meshgrid(t[:,0], x[:,0])
    tmesh_rav = tmesh.ravel().reshape(-1,1)
    xmesh_rav = xmesh.ravel().reshape(-1,1)
    Xmesh = np.concatenate([np.ones(tmesh_rav.shape), xmesh_rav, tmesh_rav], axis=1)
    g_dnn_ag_mesh = trial_func(xmesh_rav, tmesh_rav,P, L=L)
    g_analytical_mesh = np.sin(np.pi*xmesh/L)*np.exp(-(np.pi/L)**2*tmesh)
    plt.figure(figsize=(10,10))
    plt.title('Performance of neural network solving an PDE compared to the analytical solution')
    plt.plot(t, g_analytical)
    plt.plot(t, g_dnn_ag)
    plt.xlabel('t')
    plt.ylabel('g(t)')
    plt.show()

    fig, ax = plt.subplots(1,2, figsize=(10,5), subplot_kw = {'projection':'3d'})
    ax[0].plot_surface(tmesh, xmesh, g_analytical_mesh, cmap=plt.cm.coolwarm)
    ax[1].plot_surface(tmesh, xmesh, g_dnn_ag_mesh.reshape(xmesh.shape), cmap=plt.cm.coolwarm)
    ax[0].set_title('analytical')
    ax[1].set_title('NN')
    [a.set_xlabel("t") for a in ax]
    [a.set_ylabel("x") for a in ax]
    [a.set_zlabel("u(x,t)") for a in ax]

    plt.show()
