# a simple neural network
from project3.optimization import *
import matplotlib.pyplot as plt
from project3.optimization import *

import torch
from torch.autograd import grad
from torch.autograd.functional import jacobian, hessian
import torch.nn.functional as F
import torch.optim as optim

class NN_torch(torch.nn.Module):

    def __init__(self,n_input,n_hidden_neurons):
        """
        For now only logistic AF between hidden layers
        """
        super(NN_torch, self).__init__()
        self.n_hidden_neurons = n_hidden_neurons
        self.nlayers = len(n_hidden_neurons)
        self.n_input = n_input
        self.layer0 = torch.nn.Linear(3,n_hidden_neurons[0])
        for l in range(1,self.nlayers):
            # using exec to have a user defined network 
            # and that the NN class is flexible for new architectures
            exec("self.layer%i = torch.nn.Linear(n_hidden_neurons[l-1],n_hidden_neurons[l] )"%l)
    # for more flexibility I can add activation functions as an input to the forward function
    def forward(self, x):
        x = self.layer0(x)
        xd = {}
        for l in range(1,self.nlayers):
            x = torch.tanh(x)
            # using exec for the same reason as above
            exec("xd[%i] = self.layer%i(x)"%(l,l) )
            x = xd[l]
        return x

def trial_func(x, t, nn, L=torch.tensor(1)):
    """
    assume that x and t are column vectors
    """
    u0x = torch.sin(torch.pi*x/L)
    X = torch.cat((torch.ones(x.shape), x, t), 1)
    return (1-t)*u0x + t*x*(L-x)*nn.forward(X).sum(dim=1,keepdim=True)

def analytical_sol(x, t, L):
    return torch.sin(torch.pi*x/L)*torch.exp(-(torch.pi/L).pow(2)*t)

def analytical_grad(x, t, L):
    return -(torch.pi/L).pow(2)*analytical_sol(x, t)

def cost_func_pde(X, nn, L=torch.tensor(1.) ):
    # cannot use egrad because the jacobian is not diagonal
    # egrad returns the sum of each column of the jacobian
    # evaluation of the deriv trial function wrt X
    x = X[:,1].reshape(-1,1)
    t = X[:,2].reshape(-1,1)
    gtrial = trial_func(x, t, nn, L)
    # vector with same shape as columns of X
    du_dt = grad(gtrial.sum(), t, create_graph=True)[0]
    # computes the hessian with shape of the concatenated inputs 
    # (i.e. the first row of the Hessian on vector format)
    H = hessian(lambda x: trial_func(x, t, nn, L).sum(), x, create_graph=True)
    d2u_dx2 = torch.diag(H[:,0,:,0]).reshape(-1,1)
    print("du_dt.shape, d2u_dx2.shape: ", du_dt.shape, d2u_dx2.shape)
    

    # computing the cost
    cost = (d2u_dx2 - du_dt).pow(2).sum()/du_dt.shape[0]
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
    nn = NN_torch(X.shape[0], n_hidden_neurons)
    optimizer = optim.SGD(nn.parameters(), lr = eta)
    #optimizer = optim.Adam(nn.parameters(), lr = eta)
    for it in range(max_iter):
        # compute the gradient of the cost function of each layer wrt parameters P
        cost = cost_func_pde(X, nn, L=L)
        print("l2 norm of cost function", cost.pow(2).sum().pow(.5) )
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
    return nn

if __name__ == '__main__':
    np.random.seed(4155)
    # NB! Now the solver only works for same dimension along time and spatial axes
    ## Decide the vales of arguments to the function to solve
    Nt = 20
    Nx = 20
    T = torch.tensor(1.)
    L = torch.tensor(1.)
    t = torch.linspace(0, T, Nt, requires_grad=True)
    x = torch.linspace(0, L, Nx, requires_grad=True)
    tmesh, xmesh = torch.meshgrid(t, x)
    tmesh_rav = tmesh.ravel().reshape(-1,1)
    xmesh_rav = xmesh.ravel().reshape(-1,1)
    Xmesh = torch.cat([torch.ones(tmesh_rav.shape), xmesh_rav, tmesh_rav], axis=1)
    # Set up the initial parameters
    num_hidden_neurons = [20,20, len(t)]
    nlayers = len(num_hidden_neurons)
    num_iter = int(1e3)
    eta = 4e-3
    #afs = ['sigmoid']*nlayers
    nn = solve_ode_neural_network(Xmesh, num_hidden_neurons, num_iter, eta, L=L)
    # trial func and analytical solution
    g_dnn_ag_mesh = trial_func(xmesh_rav, tmesh_rav,nn, L=L).reshape(xmesh.shape)
    g_analytical_mesh = analytical_sol(xmesh, tmesh, L)
    # Find the maximum absolute difference between the solutons:
    diff_ag = torch.max(torch.abs(g_dnn_ag_mesh - g_analytical_mesh))
    print("The max absolute difference between the solutions is: %g"%diff_ag)
    # slice plot at a given index for t and x
    fig0, ax0 = plt.subplots(1, 2,figsize=(10,5))
    fig0.suptitle('Performance of neural network solving an PDE compared to the analytical solution')
    iplot =int( (Nx + Nt)*.25)
    ax0[0].plot(tmesh[:,iplot].detach().numpy(), g_analytical_mesh[:,iplot].detach().numpy(), c='k', label='analytical')
    ax0[0].plot(tmesh[:,iplot].detach().numpy(), g_dnn_ag_mesh[:,iplot].detach().numpy(), c='r', label='NN')
    ax0[1].plot(xmesh[iplot].detach().numpy(), g_analytical_mesh[iplot].detach().numpy(), c='k', label='analytical')
    ax0[1].plot(xmesh[iplot].detach().numpy(), g_dnn_ag_mesh[iplot].detach().numpy(), c='r', label='NN')
    ax0[0].set_xlabel('t')
    ax0[1].set_xlabel('x')
    ax0[0].set_ylabel('u(0,t)')
    ax0[1].set_ylabel('u(x,0)')
    [axis.legend() for axis in ax0]
    # mesh 3D plot
    fig, ax = plt.subplots(1,2, figsize=(10,5), subplot_kw = {'projection':'3d'})
    ax[0].plot_surface(tmesh.detach().numpy(), xmesh.detach().numpy(), g_analytical_mesh.detach().numpy(), cmap=plt.cm.coolwarm)
    ax[0].plot_surface(tmesh.detach().numpy(), xmesh.detach().numpy(), g_dnn_ag_mesh.detach().numpy(), cmap=plt.cm.spring, alpha=0.4)
    ax[1].plot_surface(tmesh.detach().numpy(), xmesh.detach().numpy(), g_dnn_ag_mesh.detach().numpy(), cmap=plt.cm.coolwarm)
    # plot the lines from the previous slice
    for a in ax:
        a.plot(tmesh[:,iplot].detach().numpy(), xmesh[:,iplot].detach().numpy(), g_analytical_mesh[:,iplot].detach().numpy(), c='k', label='analytical', lw=3)
        a.plot(tmesh[:,iplot].detach().numpy(), xmesh[:,iplot].detach().numpy(), g_dnn_ag_mesh[:,iplot].detach().numpy(),     c='r', label='NN',         lw=3)
        a.plot(tmesh[iplot].detach().numpy(),   xmesh[iplot].detach().numpy(),   g_analytical_mesh[iplot].detach().numpy(),   c='k', label='analytical', lw=3)
        a.plot(tmesh[iplot].detach().numpy(),   xmesh[iplot].detach().numpy(),   g_dnn_ag_mesh[iplot].detach().numpy(),       c='r', label='NN',         lw=3)

    ax[0].set_title('analytical')
    ax[1].set_title('NN')
    [a.set_xlabel("t") for a in ax]
    [a.set_ylabel("x") for a in ax]
    [a.set_zlabel("u(x,t)") for a in ax]

    plt.show()
