# to test the NN on a classification problem
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.preprocessing import scale

def create_X(x, n ):
    N = len(x)
    X = np.ones((N,n+1))
    # make the designmatrix if two independent variables
    for i in range(1,n+1):
            X[:,i] = x**i
    return X


if __name__ == "__main__":
    # for reproducability
    np.random.seed(155)
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
    # init dicts
    mse = {}
    bias= {}
    var = {}

    # true noise that was added
    sigma2 = .1
    lmbda =1e-2 #Define hyperparameter
    # scale the datasets
    X_train, X_test, Y_train, Y_test = scale(X_train), scale(X_test), scale(Y_train), scale(Y_test)
    B = 400
    polydegs = np.arange(15) + 1
    for pdeg in range(1, polydegs[-1]+1, 1):
        X = create_X(X_train[:,0], pdeg)
        n = X_train.shape[0]
        XT_X = X.T @ X
        #Ridge parameter lambda
        Id = n * lmbda * np.eye(XT_X.shape[0])
        beta_ridge = np.linalg.inv(XT_X+Id) @ X.T @ Y_train
        Xt = create_X(X_test[:,0], pdeg)
        ypred = np.sum(Xt @ beta_ridge, axis=1, keepdims=True)
        #print("ypred shape", ypred.shape)
        ystar = np.zeros((B, len(ypred) ) )
        for b in range(B):
            rinds = np.random.randint(len(ypred),size=len(ypred))
            ystar[b] = ypred[rinds,0]
        # compute the bootstrap estimate of the prediction
        ystar_bar = np.mean(ystar, axis=0, keepdims = True).T
        
        #print("ypred shape", ypred.shape)
        #print("ystar shape", ystar.shape)
        #print("ystar_bar shape", ystar_bar.shape)
        #print("Y_test shape", Y_test.shape)
        # computes mse, var, bias
        mse[pdeg] =  np.mean( (Y_test - ypred)**2, axis=0)
        bias[pdeg] = np.mean( (Y_test - ystar_bar)**2)
        var[pdeg] =  np.mean(np.var(ystar, axis=0))
        plotridge = False
        if plotridge == True:
            plt.figure()
            plt.scatter(X_test, Y_test, label='true')
            plt.scatter(X_test, ypred, label='ridge')
            plt.legend()
            plt.show()
    # plot of the mse, bias, variance
    fig = plt.figure()
    #print("shape var and bias resp", np.array(list(var.values()) ).shape, np.array(list(bias.values()) ).shape)
    plt.plot( polydegs, np.array(list(mse.values())) ,   label='mse', marker='o')
    plt.plot( polydegs, np.array(list(var.values()) ),   label='var', marker='^' )
    plt.plot( polydegs, np.array(list(bias.values())), label='bias', marker='d')
    plt.plot( polydegs, np.array(list(var.values()) ) + np.array(list(bias.values()) ) + sigma2,
              label='var + bias + $\\sigma^2$', marker='x')
    plt.xlabel(" Polynomial degree")
    plt.legend()
    plt.show()
    fig.savefig(os.getcwd() + "/../bias_var_to/bv_to_ridge.png", dpi=150)

                                                               
