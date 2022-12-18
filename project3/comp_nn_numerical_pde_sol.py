# comparison of numerical and nn pde solver of heat equation with focus on timestep

# imports from previous projects
from project2.plot import plot_heatmap
# imports for this project
from project3.optimization import *
from project3.simple_pde_neural_network import *
from project3.numerical_sol_pde import *
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
    """

    NUMERICAL SOLUTION

    """
    L = 1
    T = 1
    alpha = 0.4
    dxs = [1e-2,5e-2,1e-1]
    dts = [dx**2*alpha for dx in dxs]
    errors = []
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    for i in range(len(dxs)):
        # assigning space- and timestep
        dx = dxs[i]
        dt = dts[i]
        # defining the number of points for the time and space mesh
        # +1 to include zero
        Nt = int(np.round(L/dt))+1
        Nx = int(np.round(T/dx))+1
        # generating space and time mesh
        x = np.linspace(0,L, Nx)
        t = np.linspace(0,T, Nt)
        # ensure correct temporal and spatial resolution in the mesh
        assert np.isclose(dt, t[1]-t[0], atol=1e-10), print("dt and t[1]-t[0]", dt, t[1]-t[0])
        assert np.isclose(dx, x[1]-x[0], atol=1e-10), print("dx and x[1]-x[0]", dx, x[1]-x[0])
        xmesh, tmesh = np.meshgrid(x, t)
        # init condition
        V0 = np.sin(np.pi*x/L).reshape(-1,1)
        # num sol
        V, l2_last = solve_pde_explicit_euler(V0, dx, dt, Nx, Nt)
        # analytical solution
        Vanal = analytical_sol(xmesh, tmesh, L)
        # mean error
        eps_mean = np.mean((V[:,:,0]-Vanal)**2)**.5
        errors.append(eps_mean)
        # to visually confirm right shape of solution
        plotsurf=False
        if plotsurf:
            fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
            ax.plot_surface(xmesh, tmesh, V[:,:,0], cmap=plt.cm.coolwarm)
            ax.plot_surface(xmesh, tmesh, Vanal,    cmap=plt.cm.spring, alpha=0.4)
            plt.show()
        # exponential regime
        ax[0].plot(x, V[Nt//4,:,0], label='numerical $\\Delta x = 10^{%.2f},\\Delta t = 10^{%.2f}$'%(np.log10(dx), np.log10(dt) ) )
        ax[0].plot(x, Vanal[Nt//4,:], label='analytical $\\Delta x = 10^{%.2f},\\Delta t = 10^{%.2f}$'%(np.log10(dx), np.log10(dt) ))
        # linear regime
        ax[1].plot(x, V[3*Nt//4,:,0], label='numerical $\\Delta x = 10^{%.2f},\\Delta t = 10^{%.2f}$'%(np.log10(dx), np.log10(dt) ))
        ax[1].plot(x, Vanal[3*Nt//4,:], label='analytical $\\Delta x = 10^{%.2f},\\Delta t = 10^{%.2f}$'%(np.log10(dx), np.log10(dt) ))
        [a.legend() for a in ax]
    plt.show()
    fig, ax = plt.subplots(1,2,figsize=(10,5), sharey=True)
    dtlin = np.logspace(np.log10(min(dts)), np.log10(max(dts)),10)
    dxlin = np.logspace(np.log10(min(dxs)), np.log10(max(dxs)),10)
    ax[0].plot(dxs, errors, label="$\\epsilon$", ls='--', marker = 'o')
    ax[0].plot(dxlin, dxlin**2, label="$\\Delta x^2$")
    ax[1].plot(dts, errors, label="$\\epsilon$", ls='--', marker = 'o')
    ax[1].plot(dtlin, dtlin, label="$\\Delta t$")
    for a in ax:
        a.plot(np.array(dts)/np.array(dxs)**2, errors, label="$\\epsilon (\\alpha)$")
        a.set_xscale('log')
        a.set_yscale('log')
        a.legend()
    plt.show()

    """
    
    NEURAL PDE SOLVER

    """

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
    eta = 4e-3
    lmbda = torch.tensor(1e-4 )
    nn = solve_pde_neural_network(Xmesh_train, num_hidden_neurons, num_iter, eta, L=L)
    # trial func and analytical solution
    xmesh_test = Xmesh_test[:,1].ravel().reshape(-1,1)
    tmesh_test = Xmesh_test[:,2].ravel().reshape(-1,1)
    g_dnn_ag_mesh = trial_func(xmesh_test, tmesh_test,nn, L=L)
    g_analytical_mesh = analytical_sol(xmesh_test, tmesh_test, L)
