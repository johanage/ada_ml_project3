# generating the regression data needed for extra points
import numpy as np
from project2.test import test_func_poly_deg_p
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import scale
if __name__ == '__main__':
    # for reproducability
    np.random.seed(3155)

    # gernerate the simple 2deg poly
    x = np.linspace(-10,10,1000)
    #y = test_func_poly_deg_p(deg = 4, avec = [1,2, 4, 5, 3], x = x)
    y = np.exp(-x) + np.exp(x)
    sigma2 = .1
    ynoisy = y + np.random.normal(0,sigma2,size=y.shape)
    #x = (x - np.mean(x))/np.std(x) # pre-processing of input
    X = x[:,np.newaxis]
    Y = ynoisy[:,np.newaxis]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 3155)
    X_train, X_test, Y_train, Y_test = scale(X_train), scale(X_test), scale(Y_train), scale(Y_test)

    dir_rd = os.getcwd() + "/regdata/"
    with open(dir_rd + 'Xtrain.npy', 'wb') as f:
        np.save(f, X_train )
    f.close()

    with open(dir_rd + 'Xtest.npy', 'wb') as f:
        np.save(f, X_test )
    f.close()

    with open(dir_rd + 'Ytrain.npy', 'wb') as f:
        np.save(f, Y_train )
    f.close()

    with open(dir_rd + 'Ytest.npy', 'wb') as f:
        np.save(f, Y_test )
    f.close()
