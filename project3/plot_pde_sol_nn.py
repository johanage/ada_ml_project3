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
    num_hidden_neurons = n_hidden_neurons[-1][1]
    nlayers = len(num_hidden_neurons)
    num_iter = 200
    etas = [8e-4, 1e-3, 4e-3, 8e-3, 1e-2]
    lambdas = torch.tensor(np.logspace(-4,-1,4) )
    mses = torch.zeros((len(etas), len(lambdas)))
    for i in range(len(etas)):
        eta = etas[i]
        for j in range(len(lambdas)):
            lmbda = lambdas[j]
            nn = solve_pde_neural_network(Xmesh_train, num_hidden_neurons, num_iter, eta, L=L)
            # trial func and analytical solution
            xmesh_test = Xmesh_test[:,1].ravel().reshape(-1,1) 
            tmesh_test = Xmesh_test[:,2].ravel().reshape(-1,1) 
            g_dnn_ag_mesh = trial_func(xmesh_test, tmesh_test,nn, L=L)
            g_analytical_mesh = analytical_sol(xmesh_test, tmesh_test, L)
            mse = (g_dnn_ag_mesh - g_analytical_mesh).pow(2).mean()
            print("MSE: ", mse)
            mses[i,j] = mse
    mses = mses.detach().numpy()
    plot_heatmap(lambdas.detach().numpy(),etas,mses, title = 'Test MSE', type_axis = 'float',
                 xlabel="$\\lambda$", ylabel="$\\eta$", cbar_label = 'MSE', vmin = np.min(mses), vmax=np.max(mses),
                 store = True, store_dir = os.getcwd() + "/../plots/")
