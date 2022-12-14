# all the code for the neural network
# column vectors are by definition NOT transpose
# row vectors are transpose such that it follow the theoretical defnition Matrix (n,m) @ Column vector (m,1) = Column vector (n,1)
from autograd import grad
import numpy as np
import autograd.numpy as np
from project3.optimization import *
import matplotlib.pyplot as plt
from project1.project1 import R2score
from sklearn.model_selection import train_test_split
from project3.cost_funcs import *

class Neural_Network(object):
    def __init__(self, X, y, costfunc, eta, def_costfunc = None, def_grad_costfunc = None, symbolic_differentiation = False, 
                 optimizer = None, devsize = 0.1, rs = 3155,
                 method = 'grad_desc_wmom', w_mom = False, beta1 = 0.9, beta2 = 0.99, gamma = 0.9, delta = 1e-8):
        """
        The initialization of the NN.

        Args:
        X - ndarray, the input data (nsamples, nfeatures)
        y - ndarray, the target data
        costfunc - str, specification of type of cost function 
        eta - float, learning rate
        specify_grad - bool, compute the gradient of the cost function from symbolic differentition
        """
        X, X_dev, y, y_dev = train_test_split(X, y, test_size = devsize, random_state = rs)
        self.target = y
        self.Xdata_full = X
        self.Ydata_full = y
        self.Xdata_dev = X_dev
        self.Ydata_dev = y_dev
        self.layers = 0
        self.nodes = {0 : self.Xdata_full.shape[1]}
        self.afs = {}
        # set costfunc
        self.costfunc = costfunc
        self.def_costfunc = def_costfunc
        self.def_grad_costfunc = def_grad_costfunc
        self.cost()
        if symbolic_differentiation:
            self.grad_cost_specific()
        else:
            self.grad_cost()
        self.a = {0 : X}
        self.weights = {}
        self.bias = {}
        self.z = {}
        # init the derivatives
        self.nabla_w_C = {}
        self.nabla_b_C = {}
        self.eta = eta
        # init optimizer
        self.method = method
        self.optimizer = optimizer

    def info(self):
        print("nodes ", [(k,v) for k,v in self.nodes.items()])
        print("activation functions ", [(k,v) for k,v in self.afs.items()])
        print("activation output key value.shape ", [(k,v.shape) for k,v in self.a.items()])
        print("weights key value.shape ", [(k,v.shape) for k,v in self.weights.items()])
        print("bias key value.shape ", [(k,v.shape) for k,v in self.bias.items()])
        print("z key value.shape ", [(k,v.shape) for k,v in self.z.items()])
        print("delta key value.shape ", [(k,v.shape) for k,v in self.delta.items()])
        print("nabla_b_C key value.shape ", [(k,v.shape) for k,v in self.nabla_b_C.items()])
        print("nabla_w_C key value.shape ", [(k,v.shape) for k,v in self.nabla_w_C.items()])
    
    def add_layer(self, nodes, af, weights = None, bias = None):
        """
        To add a layer to the NN.
        Args:
        nodes - int, number of nodes in the new layer
        af - str, the activation function type
        weights - ndarray float, init weights; if not given it will be init with ~N(0,.5) 
        bias - ndarray float, init bias; if not given it will be init with ~N(0,.5)
        """
        self.layers += 1
        l = self.layers
        self.nodes[l] = nodes
        self.afs[l] = af 
        self.scale = 1
        #if af == 'sigmoid':
        #    self.scale = np.sqrt(2/(self.a[0].shape[0] + self.target.shape[0]) )
        #else:
        #    self.scale = np.sqrt(6/(self.a[0].shape[0] + self.target.shape[0]) )
        print("Variance of init weihgts: ", self.scale)
        if weights is not None: self.weights[l] = weights
        else: self.weights[l] = np.random.normal(0,self.scale, size=(self.nodes[l-1], self.nodes[l] ) )
        if bias is not None: self.bias[l] = bias
        else: self.bias[l] = np.zeros((self.nodes[l],1) ) + 0.1

    def output_layer(self, af, weights = None, bias = None):
        """
        Adds output layer to the NN for regression type problems where the output shape is equal to the input shape.
        Args:
        nodes - int, number of nodes in the new layer
        af - str, the activation function type
        weights - ndarray float, init weights; if not given it will be init with ~N(0,.5) 
        bias - ndarray float, init bias; if not given it will be init with ~N(0,.5)
        """
        self.layers += 1
        l = self.layers
        self.nodes[l] = self.nodes[0]
        self.afs[l] = af
        if weights is not None: self.weights[l] = weights
        else: self.weights[l] = np.random.normal(0,self.scale, size=(self.nodes[l-1], self.nodes[l] ) )
        if bias is not None: self.bias[l] = bias
        else: self.bias[l] = np.zeros( (self.nodes[l], 1) ) + 0.1

    def cost(self):
        """
        Computes the gradient of the cost function wrt
        the output activations at layer l
        """
        if self.costfunc == 'ols':
            self.C = cost_ols
        if self.costfunc == 'ridge':
            self.C = cost_ridge
        if self.costfunc == 'lasso':
            self.C = cost_lasso
        if self.costfunc == 'cross_entropy':
            self.C = cross_entropy
        if self.costfunc == 'cross_entropy_l1reg':
            self.C = cross_entropy_l1reg
        if self.costfunc == 'cross_entropy_l2reg':
            self.C = cross_entropy_l2reg
        if self.costfunc == "ode":
            self.C = self.def_costfunc

    def grad_cost_specific(self):
        """
        Computes the gradient of the cost function 
        using symbolic differentiation wrt the 
        output activations at layer l
        """
        if self.costfunc == "ols":
            self.nabla_a_C = grad_ols
        if self.costfunc == 'ridge':
            self.nabla_a_C = grad_ridge
        if self.costfunc == 'lasso':
            self.nabla_a_C = grad_lasso
        if self.costfunc == "cross_entropy":
            self.nabla_a_C = grad_cross_entropy
        if self.costfunc == "cross_entropy_l2reg":
            self.nabla_a_C = grad_cross_entropy_l2reg
        if self.costfunc == "cross_entropy_l1reg":
            self.nabla_a_C = grad_cross_entropy_l1reg
        if self.costfunc == "ode":
            self.nabla_a_C = self.def_grad_costfunc
    
    def grad_cost(self):
        """
        Computes the gradient of the cost function using autograd
        wrt the output activations at layer l
        """
        if self.costfunc == 'ols':            
            self.nabla_a_C = grad(cost_ols, 1)
        if self.costfunc == 'ridge':    
            self.nabla_a_C = grad(cost_ridge, 1)
        if self.costfunc == 'lasso':
            self.nabla_a_C = grad(cost_lasso, 1)
        if self.costfunc == 'cross_entropy':
            self.nabla_a_C = grad(cross_entropy, 1)
        if self.costfunc == 'cross_entropy_l1reg':
            self.nabla_a_C = grad(cross_entropy_l1reg, 1)
        if self.costfunc == 'cross_entropy_l2reg':
            self.nabla_a_C = grad(cross_entropy_l2reg, 1)
    
    def feed_forward(self):
        """
        Feed forward algorithm for all layers.
        """
        for l in range(1, self.layers+1):
            #print("self.a[l-1].shape, self.weights[l].shape, self.bias[l].T.shape", self.a[l-1].shape, self.weights[l].shape, self.bias[l].T.shape)
            self.z[l] = self.a[l-1] @ self.weights[l] + self.bias[l].T
            self.a[l] = self.sigma(self.afs[l], self.z[l]) 
    
    def predict(self, X):
        self.a[0] =  X
        self.feed_forward()
        return self.a[self.layers]

    def delta_L(self, **kwargs):
        # compute the gradient for the last layer to start backpropagation
        if self.costfunc == 'cross_entropy':
            dL = grad_ols(self.target, self.a[self.layers], **kwargs) # cross-entropy and ols have same simplified form of the gradient
        else:
            if self.costfunc == 'ols':
                gradient = self.nabla_a_C(self.target, self.a[self.layers], **kwargs)
            elif self.costfunc == 'ode':
                P = [self.bias, self.weights] # P = [bias^L, weights^L]
                gradP = self.nabla_a_C(self.Xdata_full, P, self, kwargs['init_cond'], kwargs['alpha'])
                gradient = self.sigma(self.afs[self.layers], self.a[self.layers-1] @ gradP[1][self.layers] + gradP[0][self.layers].T)
                print("and i did get here, this is gradient \n\n", gradient)
            else:
                gradient = self.nabla_a_C(self.target, self.a[self.layers], self.weights[self.layers], **kwargs)
        if self.costfunc != 'ode': print("max gradient ", np.max(gradient))
        sigma_prime_L = self.sigma_prime(self.afs[self.layers], self.z[self.layers])
        dL = np.multiply(gradient, sigma_prime_L )
        self.delta = {self.layers : dL }

    def delta_l(self,l):
        dl = np.multiply(self.delta[l+1] @ self.weights[l+1].T, self.sigma_prime(self.afs[l], self.z[l]) )
        self.delta[l] = dl
        
    def backpropagation(self, **kwargs):
        # compute the error in layer
        self.delta_L(**kwargs)
        for l in range(self.layers):
            idx = self.layers-l
            if l > 0:
                self.delta_l(idx)
            self.nabla_b_C[idx] = np.sum(self.delta[idx], axis = 0, keepdims=True)
            self.nabla_w_C[idx] = self.a[idx-1].T @ self.delta[idx]
    
    def update_weights(self, size_mini_batches, method, count = None):
        for l in range(1,self.layers+1):
            # update bias
            self.optimizer.theta = self.bias[l]
            self.optimizer.mt = np.zeros(self.optimizer.theta.shape)
            self.optimizer.vt = np.zeros(self.optimizer.theta.shape)
            self.optimizer(self.method, gradients_in = self.nabla_b_C[l].T, max_iter = 1)
            self.bias[l] = np.sum([-self.bias[l], self.optimizer.theta], axis=1)
            
            print("right before weight update")
            self.optimizer.theta = self.weights[l]
            self.optimizer.mt = np.zeros(self.optimizer.theta.shape)
            self.optimizer.vt = np.zeros(self.optimizer.theta.shape)
            self.optimizer(self.method, gradients_in = self.nabla_w_C[l], max_iter = 1)
            self.weights[l] -= self.optimizer.theta


    def SGD(self, epochs, size_mini_batches, method,tol=1e-4, printout = False, plot=False, 
            store_grads = False, store_activation_output = False, batchnorm = False, **kwargs):
        """
        Stochastic gradient descent for optimizin the weights and biases in the NN.
        Tip: choose number of minibatches s.t. the lenth of each batch is a power of 2.
        
        Args:
        - epochs            - int, number of epochs
        - size_mini_batches - int, number of samples in each mini batch (approx)
        - printout          - bool, prints out information on loss wrt to epoch number and batch number
        - plot              - bool, plots the (epoch number, loss)
        """
        nsamples = self.Ydata_full.shape[0]
        mini_batches = nsamples//size_mini_batches
        print(" # mini batches", mini_batches)
        self.r2 = np.zeros(epochs)
        self.mse = np.zeros(epochs)
        self.losses = np.zeros(epochs)
        self.losses_dev = np.zeros(epochs)
        self.l2norm_gradC_weights = np.ones(epochs)*np.nan
        # storing norm of gradients for analysing convergence
        if store_grads:
            self.grads = {}
            self.grads[0] = self.nabla_w_C
        # storing activation output for each layer at each iteration
        if store_activation_output:
            self.activation_output = {}
            self.activation_output[0] = self.a
        for nepoch in range(epochs):
            bias_old = self.bias[self.layers]
            weights_old = self.weights[self.layers]
            for m in range(mini_batches):
                count = nepoch*mini_batches + m + 1
                ind_batch = np.random.randint(nsamples - size_mini_batches)
                binds = divide_batches(X = self.Xdata_full, nmb = mini_batches, istart = ind_batch)
                # bactch normalization
                if batchnorm:
                    data = ( self.Xdata_full[binds]- np.sum(self.Xdata_full[binds], axis = 0)/len(binds) )/np.std(self.Xdata_full[binds], axis = 0)
                else: 
                    data = self.Xdata_full[binds]
                # set the first activation output and target to the batch samples 
                self.a[0] = data 
                self.target = self.Ydata_full[binds]
                # do feed forward on the batch
                self.feed_forward()
                # do backprop on the batch
                self.backpropagation(**kwargs)
                # update weights and biases
                self.update_weights(size_mini_batches, method, count)
                # store weights and activation output for anlaysis
                if store_grads:
                    self.grads[count] = self.nabla_w_C
                if store_activation_output:
                    self.activation_output[count] = self.a
                # print information
                if printout:
                    print("max of activation output: ", np.max(self.a[self.layers-1]))
                    print("Epoch {0}/{1}, batch {2}/{3}, loss: 1e{4:.4f}".format(nepoch+1,epochs,m+1,mini_batches, np.log10(np.mean(self.losses))) )
                    print("Max delta weights: ", np.max(self.weights[self.layers] - weights_old))
                    print("Max delta bias: ", np.max(self.bias[self.layers] - bias_old))
            # compute loss to indicate performance wrt epoch number
            if self.costfunc in ['ols','cross_entropy']:
                loss = self.C(self.Ydata_full, self.predict(self.Xdata_full), **kwargs)
                loss_dev = self.C(self.Ydata_dev, self.predict(self.Xdata_dev), **kwargs)
            else:
                loss = self.C(self.Ydata_full, self.predict(self.Xdata_full), self.weights[self.layers], **kwargs)
                loss_dev = self.C(self.Ydata_dev, self.predict(self.Xdata_dev), self.weights[self.layers], **kwargs)
            self.losses[nepoch] = loss
            self.losses_dev[nepoch] = loss_dev
            self.r2[nepoch] = 1 - np.sum(np.sum((self.Ydata_full - self.predict(self.Xdata_full))**2, axis=0) / np.sum( (self.Ydata_full - np.sum(self.Ydata_full, axis=0)/self.target[0])**2 ) )#R2score(self.target, self.a[self.layers])
            self.mse[nepoch] = cost_ols(self.Ydata_full, self.predict(self.Xdata_full), **kwargs)
            self.l2norm_gradC_weights[nepoch] = np.sqrt(np.sum((self.nabla_w_C[self.layers])**2))
            if nepoch > 0 and self.l2norm_gradC_weights[nepoch] <=tol: self.bconv = True; break 
            if printout:
                print("Epoch {0}/{1}, l2-norm grad C_weights: 1e{2:.4f}".format(nepoch+1,epochs, np.log10(np.mean(self.l2norm_gradC_weights))) )
        if plot:
            plt.plot(np.arange(epochs) + 1, self.losses)
            plt.xlabel("Epochs")
            plt.ylabel("$C(\\theta)$")
            plt.show()

    def sigma(self, af, zl):
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
           
    def sigma_prime(self, af, zl, delta = None):
        """
        Args:
        af - string, activation function
        zl - ndarray, neurons weighted input at layer l
        """
        if af == "linear":
            out = 1
        if af == "sigmoid":
            s = self.sigma(af, zl)
            out = s*(1 - s)
        if af == "tanh":
            out = 1/np.cosh(zl)
        if af == "relu":
            out = .5*(1 + np.sign(zl))
        if af == "leaky_relu":
            out = .5*( (1 - np.sign(zl))*1e-2 + (1 + np.sign(zl)))
        if af == "softmax":
            s = self.sigma(af, zl)
            out = s*(1 - s)
        return out
