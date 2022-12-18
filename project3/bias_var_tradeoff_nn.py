# to test the NN on a classification problem
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.model_selection import train_test_split
# impor from personal projects
from project2.neural_network import *
from project1.project1 import R2score, MSE


if __name__ == "__main__":
    # for reproducability
    np.random.seed(3155)
    with open(os.getcwd() + '/regdata/Xtrain.npy', 'rb') as f:
        X_train = np.load(f)
    f.close()
    with open(os.getcwd() + '/regdata/Xtest.npy', 'rb') as f:
        X_test = np.load(f)
    f.close()
    with open(os.getcwd() + '/regdata/Ytrain.npy', 'rb') as f:
        Y_train = np.load(f)
    f.close()   
    with open(os.getcwd() + '/regdata/Ytest.npy', 'rb') as f:
        Y_test = np.load(f)
    f.close()
    # true noise that was added
    sigma2 = .1
     
    # scale the data (same as sklearn preprocessing scale function)
    X_train = (X_train - np.mean(X_train))/np.std(X_train)
    X_test  = (X_test -  np.mean(X_test)) /np.std(X_test)
    Y_train = (Y_train - np.mean(Y_train))/np.std(Y_train)
    Y_test  = (Y_test -  np.mean(Y_test)) /np.std(Y_test)
    X_train = np.concatenate( (np.ones(X_train.shape), X_train), axis=1)
    X_test = np.concatenate( (np.ones(X_test.shape), X_test), axis=1)

    # define hyperparameters
    eta=    1e-3                   #Define vector of learning rates (parameter to SGD optimiser)
    lmbda = 1e-5                                  #Define hyperparameter
    n_hidden_neurons = [2, 5, 10, 20, 50, 100, 500, 1000]

    epochs= 200                                 #Number of reiterations over the input data
    batch_size= 10                              #Number of samples per gradient update
     
    # MSE
    Test_mse_own  = np.zeros((len(n_hidden_neurons)))      #of learning rate and number of hidden neurons for 
    # variance
    Test_var_own  = np.zeros((len(n_hidden_neurons)))      #of learning rate and number of hidden neurons for 
    # bias
    Test_bias_own  = np.zeros((len(n_hidden_neurons)))      #of learning rate and number of hidden neurons for 

    method = 'sgd'
    w_mom = True
    beta1 = 0.9
    beta2 = 0.99
    af = 'sigmoid'
    i = 0
    for neurons in n_hidden_neurons:
        nn = Neural_Network(X_train, Y_train, costfunc = 'ridge', eta=eta, w_mom = w_mom, beta1 = beta1, beta2 = beta2, method = method, symbolic_differentiation = False)
        nn.add_layer(nodes = neurons, af = af)# weights = np.random.normal(0,.1,size = (X_train.shape[1], n_hidden_neurons) ) )        
        # output layer
        nn.output_layer(af = 'linear')#  weights = np.random.normal(0,1,size=(neurons, 2) )) 
        # do SGD
        print("Epochs: ", epochs, " # mini batch size :", batch_size)
        nn.SGD(epochs = epochs, size_mini_batches = batch_size, printout=True,**{'lambda' : lmbda})
        pred_train = nn.predict(X_train)
        pred_test = nn.predict(X_test)
        # bootstrapping
        B = 400
        ystar = np.zeros((B, len(pred_test) ) )
        for b in range(B):
            rinds = np.random.randint(len(pred_test),size=len(pred_test))
            ystar[b] = pred_test[rinds]
        ystar_bar = np.mean(ystar, axis=0, keepdims = True).T
        # compute the estimates
        Test_mse_own[i] =   np.mean((Y_test - pred_test)**2)
        Test_var_own[i] =   np.mean(np.var(ystar, axis=1))
        Test_bias_own[i] =  np.mean( (Y_test - ystar_bar )**2)
        i += 1
    store_dir = "/../bias_var_to/%s_"%method

    fig = plt.figure()
    plt.plot( n_hidden_neurons, Test_mse_own,   label='mse', marker='o')
    plt.plot( n_hidden_neurons, Test_var_own,   label='var', marker='^' )
    plt.plot( n_hidden_neurons, Test_bias_own, label='bias', marker='d')
    plt.plot( n_hidden_neurons, Test_var_own + Test_bias_own + sigma2, 
              label='var + bias + $\\sigma^2$', marker='x')
    plt.xscale('log')
    plt.xlabel(" # Hidden neurons")
    plt.legend()
    plt.show()
    fig.savefig(os.getcwd() + "/../bias_var_to/bv_to_nn.png", dpi=150)
