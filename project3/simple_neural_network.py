# a simple neural network
from project3.optimization import *
from autograd import grad
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
from project3.optimization import *
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

def simple_NN(X,P, afs):
    """
    X - design matrix, assuming it already includes bias
    y - target data
    P - NN parameters (bias and weights)
    afs - activation functions for each layer
    """
    nhidden = len(afs)
    # variable representing the
    # output of the previous layer
    almin1 = X
    for l in range(nhidden):
        wl = P[l]
        zl = almin1 @ wl.T
        al = activation_funcs(zl, afs[l])
        almin1 = al
    wL = P[-1]
    zL = almin1 @ wl
    aL = zL
    return aL

def trial_func(X, P, init_cond, afs):
    out_nn = simple_NN(X, P, afs)
    return init_cond + X[:,1][:,np.newaxis]*out_nn

def ode_rhs(gtrial, alpha):
    return alpha * gtrial

def cost_func_ode(X, P, init_cond, afs, alpha):
    # evaluation of the deriv trial function wrt X
    lhs = egrad(trial_func, 0)(X, P, init_cond, afs)

    # eval trial function
    gtrial = trial_func(X, P, init_cond, afs)

    # evaluate the rhs of the ode
    rhs = np.sum(ode_rhs(gtrial, alpha), axis =1, keepdims=True)

    # computing the cost
    cost = np.sum((lhs - rhs)**2)/rhs.shape[0]
    return cost

def solve_ode_neural_network(X, n_hidden_neurons, max_iter, eta, init_cond, afs, alpha):
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
    P[0] = np.random.normal(size=(n_hidden_neurons[0]+1, 2))
    for l in range(1, nlayers):
        P[l] = np.random.normal(size=(n_hidden_neurons[l] + 1, n_hidden_neurons[l-1]+1))
    grad_cost = grad(cost_func_ode, 1)
    optimizer_list = [None]*nlayers
    for it in range(max_iter):
        # compute the gradient of the cost function of each layer wrt parameters P
        nabla_cost = grad_cost(X, P, init_cond, afs, alpha) 
        cost = cost_func_ode(X, P, init_cond, afs, alpha)
        print("l2 norm of cost function", np.sum(cost**2)**.5)
        for l in range(nlayers):
            print("l2 norm of cost function gradient in layer %i"%l, np.sum(nabla_cost[l]**2)**.5)
            if it == 0:
                # init the optimizer for each layer (allowing more complex updates)
                optimizer_list[l] = optimizers(X, None, cost_func_ode, eta, theta_init = P[l])
            optimizer_list[l].ADAM(nabla_cost[l], it+1)
            #print("gradient descent update ", eta*nabla_cost[l])
            P[l] = optimizer_list[l].theta
            #P[l] -= eta*nabla_cost[l]
            #print(" l2 norm of parameters in layer %i"%l, np.sum(P[l]**2)**.5)
    return P

if __name__ == '__main__':
    np.random.seed(4155)

    ## Decide the vales of arguments to the function to solve
    Nt = 50
    T = 1
    t = np.linspace(0,T, Nt)
    X = np.c_[np.ones(t.shape), t]
    ## Set up the initial parameters
    num_hidden_neurons = [20, 10, len(t)]
    nlayers = len(num_hidden_neurons)
    num_iter = int(1e4)
    eta = 1e-1
    afs = ['sigmoid']*nlayers
    alpha = 3
    init_cond = np.ones(t.shape)[:,np.newaxis]
    P = solve_ode_neural_network(X, num_hidden_neurons, num_iter, eta, init_cond = init_cond, afs = afs, alpha = alpha)
    # trial func and analytical solution
    g_dnn_ag = trial_func(X,P, init_cond = init_cond, afs = afs)
    g_analytical = np.exp(alpha*X)[:,1][:,np.newaxis]

    # Find the maximum absolute difference between the solutons:
    diff_ag = np.max(np.abs(g_dnn_ag - g_analytical))
    print("The max absolute difference between the solutions is: %g"%diff_ag)

    plt.figure(figsize=(10,10))
    plt.title('Performance of neural network solving an ODE compared to the analytical solution')
    plt.plot(t, g_analytical)
    plt.plot(t, np.mean(g_dnn_ag, axis=1))
    plt.legend(['analytical','nn'])
    plt.xlabel('t')
    plt.ylabel('g(t)')

    plt.show()
