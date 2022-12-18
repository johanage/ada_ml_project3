# to test the NN on a classification problem
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import scale

# impor from personal projects
from project2.plot import plot_heatmap
from project2.neural_network import *
from project2.test import test_func_poly_deg_p
from project1.project1 import R2score, MSE


def bias(y, ytilde):
    eytilde = np.mean(ytilde)
    return (y - eytilde)**2


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
    X_train, X_test, Y_train, Y_test = scale(X_train), scale(X_test), scale(Y_train), scale(Y_test)
    X_train = np.concatenate( (np.ones(X_train.shape), X_train), axis=1)
    X_test = np.concatenate( (np.ones(X_test.shape), X_test), axis=1)

    #Decision Tree Regression
    mse = {}
    bias= {}
    var = {}
    plotscatter = False
    B = 400
    depths = (np.arange(40)+1)
    for mdepth in depths:
        dtree_reg = DecisionTreeRegressor(max_depth=mdepth)
        dtree_reg.fit(X_train, Y_train)
        ypred = dtree_reg.predict(X_test)[:,np.newaxis]
        ystar = np.zeros((B, len(ypred) ) )
        for b in range(B):
            rinds = np.random.randint(len(ypred),size=len(ypred))
            ystar[b] = ypred[rinds,0]
        ystar_bar = np.mean(ystar, axis=0, keepdims=True).T
        #print("shape ypred ", ypred.shape)
        #print("shape Y_test", Y_test.shape)
        #print("shape ystar_bar", ystar_bar.shape)
        mse[mdepth] =  np.mean( (Y_test - ypred[:,0])**2)
        bias[mdepth] = np.mean( (Y_test - ystar_bar)**2)
        var[mdepth] =  np.mean(np.var(ystar, axis=0),    axis=0)
        #print("mse shape ",  mse[mdepth].shape, 
        #      "bias shape ", bias[mdepth].shape, 
        #      "var shape ",  var[mdepth].shape)
        if plotscatter == True:
            plt.scatter(X_train[:,1]+.2, dtree_reg.predict(X_train),  label='Pred train')
            plt.scatter(X_test[:,1]+.2, ypred,  label='Prediction')
            plt.scatter(X_test[:,1], Y_test, label='True data')
            plt.legend()
            plt.show()
    # plot of the mse, bias, variance
    #print(np.array(list(var.values()) ) + np.array(list(bias.values()) )  )
    fig = plt.figure()
    plt.plot( depths, np.array(list(mse.values())) ,   label='mse', marker='o')
    plt.plot( depths, np.array(list(var.values()) ),   label='var', marker='^' )
    plt.plot( depths, np.array(list(bias.values())), label='bias', marker='d')
    plt.plot( depths, np.array(list(var.values()) ) + np.array(list(bias.values()) ) + sigma2, 
              label='var + bias + $\\sigma^2$', marker='x')
    plt.xlabel("Tree depth")
    plt.legend()
    plt.show()
    fig.savefig(os.getcwd() + "/../bias_var_to/bv_to_dtree.png", dpi=150)

