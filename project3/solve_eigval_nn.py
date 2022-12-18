# a simple neural network

# imports form this project
from project3.optimization import *
from project3.optimization import *
from project3.simple_pde_neural_network import NN_torch
# other imports
import os
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
from torch.autograd.functional import jacobian
import torch.nn.functional as F
import torch.optim as optim


#def x_t(t, nn, af = 'tanh'):
def x_t(t, nn, af = 'sigmoid'):
    #X = torch.cat((torch.ones(t.shape), t), dim=1)
    return nn.forward(t, af = af)

def trial_func(x0, t, nn):
    xt = x_t(t, nn)
    a = torch.einsum('i...,j->ij', torch.exp(-t[:,0]), x0[:,0])
    b = torch.einsum('i...,ij->ij',(1-torch.exp(-t[:,0])), xt)
    return a + b

def equil_cond(x, A):
    """
    f(x(t)) in equation 1 in Z. Yi et al.
    """
    a = torch.einsum('ij, ij, kl, il->ik', x, x, A, x)
    b = torch.einsum('ij, jk, ik, il->il', x, A, x, x)
    return a - b

def cost_func_symmat(x0, t, nn, A, lmbda = 1e-4 ):
    # eval the trial sol
    gtrial = trial_func(x0, t, nn)
    # vector with same shape as columns of X
    # grad wrt t
    J = jacobian(lambda t: trial_func(x0, t, nn).sum(dim=0), t, create_graph=True)
    dx_dt = J[:,:,0].T
    # rhs
    rhs = equil_cond(gtrial, A)
    # computing the cost
    loss = torch.nn.MSELoss()
    reg = torch.tensor(0.)
    for l in range(2,2*(nn.nlayers)+1,2):
        reg += nn.state_dict()[list(nn.state_dict().keys())[-l]].pow(2).sum()
    cost = loss(dx_dt, rhs) + lmbda*reg/t.shape[0] # t.shape[0] corresponds to number of samples
    return cost

def solve_symmat_nn(x0, t, A, n_hidden_neurons, max_iter, eta, lmbda = 1e-4):
    """
    X - design matrix
    y - target data
    P - NN parameters (bias and weights)
    n_hidden_neurons - tuple of # neurons for each layer, lenght of tuple is the # layers
    eta - float, learning rate
    """
    nlayers = len(n_hidden_neurons)
    # init params
    nn = NN_torch(t.shape[0], n_hidden_neurons, n_features=1)
    #optimizer = optim.SGD(nn.parameters(), lr = eta)
    optimizer = optim.Adam(nn.parameters(), lr = eta)
    for it in range(max_iter):
        # compute the gradient of the cost function of each layer wrt parameters P
        cost = cost_func_symmat(x0, t, nn, A=A, lmbda=lmbda)
        if it%10==0:
            print("iter %i, l2 norm of cost function"%it, cost.pow(2).sum().pow(.5) )
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
    return nn

def eigval(v, A):
    """ computets the eigenvalues 
        from eigenvector and matrix A
        where v are eigenvectors"""
    #return v.T @ A @ v / v.T @ v
    return torch.einsum('ij, jk, ik->i', v, A, v)/torch.einsum('ij, ij->i', v, v)

def explicit_euler_eigval_solver(x0, A, dt, Nt_euler):
    x = torch.ones((Nt_euler, x0.shape[0], 1))
    x[0] = x0
    for i in range(1,Nt_euler):
        dxdt = x[i-1].T @ x[i-1] * A @ x[i-1] - x[i-1].T @ A @ x[i-1] * x[i-1]#equil_cond(x[i-1], A).sum(dim=1,keepdims=True)
        x[i] = x[i-1] + dt * dxdt
    return x

if __name__ == '__main__':
    np.random.seed(5)
    # Setting up the problem
    N = 6
    Nt = 11
    T = 10
    t = torch.linspace(0,T,Nt, requires_grad=True).reshape(-1,1)
    x0 = torch.randn(N).reshape(-1,1)
    
    # for Yi et al paper problem
    #x0 = torch.tensor([[-0.2679],
    #                   [ 0.1798],
    #                   [ 0.4032],
    #                   [ 0.1847],
    #                   [ 0.8362]])
    
    x0 = x0/torch.norm(x0)
    # generating random symmetric 6x6 matrix (by order of the magistirium)
    Q = torch.randn(N,N)
    A = (Q.T + Q)/2
    
    # paper problem
    #A = torch.tensor([
    #                [ 0.7663,  0.4283, -0.3237,  -0.4298, -0.1438], 
    #                [ 0.4283,  0.2862,  0.0118,  -0.2802,  0.1230],
    #                [-0.3237,  0.0118, -0.9093,  -0.4384,  0.7684],
    #                [-0.4298, -0.2802, -0.4384,  -0.0386, -0.1315],
    #                [-0.1438,  0.1230,  0.7684,  -0.1315, -0.4480]
    #                ])
    
    # Set up the initial parameters
    lnum_hidden_neurons = [
                            [100,50,25, len(x0)],
                            [50,25, len(x0)],
                            [200, 100, 50, 25, len(x0)],
                            [10, len(x0)]
                          ]
    for num_hidden_neurons in lnum_hidden_neurons:
        nlayers = len(num_hidden_neurons)
        num_iter = 1000
        eta = 1e-3 #1e-2
        lmbda = 1e-5
        NNsol = solve_symmat_nn(x0, t, A, num_hidden_neurons, num_iter, eta, lmbda)
        xs = trial_func(x0, t, NNsol)
        # finding the  eigenvector at an equilibrium point
        Nt_eq = 10001 
        t_eq = torch.linspace(0, T, Nt_eq).reshape(-1,1)
        xis = trial_func(x0, t_eq, NNsol)
        print("shape xis ", xis.shape)
        xiT = xis[-1].reshape(-1,1)
        print("xiT\n",xiT)
        equil_test = equil_cond(xiT, A)
        print("Equilibrium test: ", equil_test)
        eigval_nn_points = eigval(xs, A)
        eigval_nn = eigval(xis, A)
        
        # following result of thm 5 in Yi et al to find the min eigenvalue
        t = torch.linspace(0,T,Nt, requires_grad=True).reshape(-1,1)
        NNsol_min = solve_symmat_nn(x0.detach(), t,-A, num_hidden_neurons, num_iter, eta, lmbda)
        t_eq = torch.linspace(0, T, Nt_eq).reshape(-1,1)
        xs_min  = trial_func(x0.detach(), t,    NNsol_min)
        xis_min = trial_func(x0.detach(), t_eq, NNsol_min)
        print("shape xis ", xis_min.shape)
        xiT_min = xis_min[-1].reshape(-1,1)
        print("xiT\n",xiT)
        equil_test_min = equil_cond(xiT_min, A)
        print("Equilibrium test: ", equil_test_min)
        eigval_nn_points_min = eigval(xs_min, A)
        eigval_nn_min = eigval(xis_min, A)

        print("Max eigval from nn ", eigval_nn[-1])
        print("Min eigval from nn ", eigval_nn_min[-1])
        
        # torch linalg eig decomp
        # skip because it has worse precisino than np for some reason
        #L, V = torch.linalg.eig(A)
        #print("V from torch.linalg.eig:\n", V)
        #print("L from torch.linalg.eig:\n", L)
        

        # explicit euler scheme
        dt = 1e-3
        Nt_euler = int(T/dt) + 1
        t_euler = np.linspace(0,T, Nt_euler).reshape(-1,1)
        # finding the max eigval
        xis_euler = explicit_euler_eigval_solver(x0, A, dt, Nt_euler)[:,:,0]
        eigval_euler = eigval(xis_euler, A) 
        # finding the min eigval
        xis_euler_min = explicit_euler_eigval_solver(x0, -A, dt, Nt_euler)[:,:,0]
        eigval_euler_min = eigval(xis_euler_min, A)

        # Print results from numpy
        w, v = np.linalg.eig(A)
        print()
        print('A =', A)
        print('x0 =', x0)
        print('Eigvec Numpy \n', v)
        print('Eigval Numpy:', w)
        print('Max Eigval Numpy', np.max(w))
        print('Min Eigval Numpy', np.min(w))
        
        # Plot eigenvalues
        fig, ax = plt.subplots()
        lgd_numpy = "Numpy $\\lambda_{\\mathrm{max}} \\sim$ " + \
            str(round(np.max(w), 5))
        lgd_numpy_min = "Numpy $\\lambda_{\\mathrm{min}} \\sim$ " + \
            str(round(np.min(w), 5))
        lgd_euler = "Euler $\\lambda_{\\mathrm{max}} \\sim$ " + \
            str(round(eigval_euler.detach().numpy()[-1], 5))
        lgd_euler_min = "Euler $\\lambda_{\\mathrm{min}} \\sim$ " + \
            str(round(eigval_euler_min.detach().numpy()[-1], 5))
        lgd_nn = "FFNN $\\lambda_{\\mathrm{max}} \\sim$ " + \
            str(round(eigval_nn.detach().numpy()[-1], 5))
        lgd_nn_min = "FFNN $\\lambda_{\\mathrm{min}} \\sim$ " + \
            str(round(eigval_nn_min.detach().numpy()[-1], 5))
        # plot the max and min eigval from np
        ax.axhline(np.max(w), color='r', ls='--', label=lgd_numpy)
        ax.axhline(np.min(w), color='C3',ls='--', label=lgd_numpy_min)
        # plot the max min eigval from neural solver
        # continuous plot and scatter points respectively
        ax.plot(t_euler[::20], eigval_euler[::20],     c='C3',  ls='--', marker = '^', ms = 2,  label=lgd_euler)
        ax.plot(t_euler[::20], eigval_euler_min[::20], c='C4',  ls='--', marker = '^', ms = 2,  label=lgd_euler_min)
        ax.plot(t_eq.detach().numpy(), eigval_nn.detach().numpy(),            c='b', label=lgd_nn)
        ax.scatter(t.detach().numpy(), eigval_nn_points.detach().numpy(),     c='C1')
        ax.plot(t_eq.detach().numpy(), eigval_nn_min.detach().numpy(),        c='c', label=lgd_nn_min)
        ax.scatter(t.detach().numpy(), eigval_nn_points_min.detach().numpy(), c='C2')
        ax.set_xlabel('Time, $t$')
        ax.set_ylabel('$\\lambda$')
        plt.legend(loc='best')
        plt.show()
        snarch = ""
        for s in str(num_hidden_neurons)[1:-1].split(","):
            snarch += s.strip() + "_"
        fn = "eigval_nnarch_%seta_1e%.2f_lambda_1e%.2f_%iiter.png"%(snarch, np.log10(eta), np.log10(lmbda), num_iter)
        print(" filename ", fn)
        fig.savefig(os.getcwd() + "/../plots/d/%s"%fn, dpi=150)
