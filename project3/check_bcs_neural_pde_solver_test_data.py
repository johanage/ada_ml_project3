# a simple neural network
# imports from previous projects
from project2.plot import plot_heatmap
# imports for this project
from project3.optimization import *
from project3.simple_pde_neural_network import *
# other imports
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
from torch.autograd.functional import jacobian, hessian
import torch.nn.functional as F
import torch.optim as optim

if __name__ == '__main__':
    np.random.seed(4155)
    # static params
    Nt = 20
    Nx = 20
    T = torch.tensor(1.)
    L = torch.tensor(1.)

    # data
    t = torch.linspace(0, T, Nt, requires_grad=True)
    x = torch.linspace(0, L, Nx, requires_grad=True)
    tmesh, xmesh = torch.meshgrid(t, x)
    tmesh_rav = tmesh.ravel().reshape(-1,1)
    xmesh_rav = xmesh.ravel().reshape(-1,1)
    Xmesh = torch.cat([torch.ones(tmesh_rav.shape), xmesh_rav, tmesh_rav], axis=1)
    npXmesh_train, npXmesh_test = train_test_split(Xmesh.detach().numpy(), test_size=0.2, random_state=42)
    Xmesh_train = torch.tensor(npXmesh_train, requires_grad=True)
    Xmesh_test  = torch.tensor(npXmesh_test,  requires_grad=True)
    # Set up the initial parameters
    n_hidden_neurons = [[(1,len(t)),  (1,1,len(t)),   (1,1,1,len(t))   ],
                       [ (5,len(t)),  (5,5,len(t)),   (5,5,5,len(t))   ],
                       [ (10,len(t)), (10,10,len(t)), (10,10,10,len(t))],
                       [ (20,len(t)), (20,20,len(t)), (20,20,20,len(t))]]
    num_hidden_neurons = n_hidden_neurons[-1][-1]
    nlayers = len(num_hidden_neurons)
    num_iter = 10
    eta = 1e-2
    lmbda = torch.tensor(1e-3)
    nn = solve_pde_neural_network(Xmesh_train, num_hidden_neurons, num_iter, eta, L=L)
    # trial func and analytical solution
    xmesh_test = Xmesh_test[:,1].ravel().reshape(-1,1) 
    tmesh_test = Xmesh_test[:,2].ravel().reshape(-1,1) 
    g_dnn_ag_mesh = trial_func(xmesh_test, tmesh_test,nn, L=L)
    g_analytical_mesh = analytical_sol(xmesh_test, tmesh_test, L)
    
    # slice plot at a given index for t and x
    fig0, ax0 = plt.subplots(1, 3,figsize=(15,5))
    fig0.suptitle('Neural PDE solver and analytical solution Dirichlet boundary conditions')
    it0 = tmesh_test == 0
    ix0 = xmesh_test == 0
    ixL = xmesh_test == L
    ax0[0].plot(xmesh_test[it0].detach().numpy(),   g_analytical_mesh[it0].detach().numpy(), c='k', label='analytical')
    ax0[0].plot(xmesh_test[it0].detach().numpy(),   g_dnn_ag_mesh[it0].detach().numpy(), c='r', label='NN')
    ax0[1].plot(tmesh_test[ix0].detach().numpy(),   g_analytical_mesh[ix0].detach().numpy(), c='k', label='analytical')
    ax0[1].plot(tmesh_test[ix0].detach().numpy(),   g_dnn_ag_mesh[ix0].detach().numpy(), c='r', label='NN')
    ax0[2].plot(tmesh_test[ixL].detach().numpy(),   g_analytical_mesh[ixL].detach().numpy(), c='k', label='analytical')
    ax0[2].plot(tmesh_test[ixL].detach().numpy(),   g_dnn_ag_mesh[ixL].detach().numpy(), c='r', label='NN')
    ax0[0].set_xlabel('x')
    ax0[1].set_xlabel('t')
    ax0[2].set_xlabel('t')
    ax0[0].set_ylabel('u(x,0)')
    ax0[1].set_ylabel('u(0,t)')
    ax0[2].set_ylabel('u(L,t)')
    [axis.legend() for axis in ax0]
    fig0.tight_layout()
    plt.show()
