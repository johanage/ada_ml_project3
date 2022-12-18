# regression tree code
import numpy as np
import matplotlib.pyplot as plt

def bisect(y,x, split_at):
    """
    bisects domain and computes the average of the 2 regions
    giving the output of the next to leaf nodes (which could be part of a branch)
    """
    
    x1 = x[:split_at]
    x2 = x[split_at:]
    y1 = y[:split_at]
    y2 = y[split_at:]
    return x1, x2, y1, y2

def SSR(y, ytilde):
    """
    Sum of squared residuals
    """
    r = y - ytilde
    return np.sum(r**2)

def find_split(y, x):
    """
    Find splits based on which split returns the smallest sum of squared residuals.
    Could easily change to another loss function.
    Residuals is the difference between the mean of the points (y's) in the regions
    after the split and the true data y.
    """
    Npoints = len(y)
    ssrs = []
    indices = np.arange(Npoints-1) + 1
    for i in range(1, Npoints):
        xpair_avg = (x[i]+x[i-1])/2
        print(xpair_avg)
        split = np.arange(len(x))[x < xpair_avg][-1]
        ssr1 = np.sum((y[:split] - np.mean(y[:split]))**2)
        ssr2 = np.sum((y[split:] - np.mean(y[split:]))**2)
        ssr = ssr1 + ssr2
        ssrs.append(ssr)
    return x[indices[np.argmin(ssrs)]]

def build_tree(y, x, max_leaves):
    x_to_split = x
    y_to_split = y
    split_at_old = 0
    tree = {}
    nleaves = 1
    while nleaves < max_leaves:
        xsplit   = find_split(y_to_split, x_to_split)        
        print(" split at x ", xsplit)
        split_at = np.arange(len(x))[x < xsplit][-1]
        x1, x2, y1, y2 = bisect(y_to_split, x_to_split, split_at)
        x_to_split = x2
        y_to_split = y2
        tree['%.4f'%(xsplit)] = (x1, y1)
        split_at_old += split_at
        nleaves += 1
        print(nleaves)
    tree['%.4f'%(xsplit)] = (x2, y2)
    return tree

def reg_tree(y, x, xreg, max_leaves):
    """
    1D regression with decision trees
    split_at - inital list/array of indices to split the domain, assumes that zero is included
    y - target data
    x - domain data
    """
    # for simplicity set a fixed number of splits
    tree = build_tree(y, x, max_leaves)
    yreg = np.ones(xreg.shape)
    prev_split = 0
    count = 1
    nsplits = len(list(tree.keys()))
    for key, val in tree.items():
        iupper = np.arange(len(xreg))[xreg < float(key)][-1] # indexing a list of natural numbers from 0 to len -1 of xreg
        ilower = np.arange(len(xreg))[xreg >= prev_split][0]  # indexing a list of natural numbers from 0 to len -1 of xreg
        prev_split = float(key)
        print("lower and upper boundary", ilower, iupper)
        w = np.mean(y[(prev_split <= x) & (x <= float(key) )])
        if count == nsplits: 
            print(" count is equal to nsplits")
            yreg[ilower:] *= w
        else: yreg[ilower:iupper+1] *= w
        count += 1
    return yreg

if __name__=="__main__":
    y = np.array([1,1,2,2,3,3])
    x = np.linspace(0,1, len(y))
    #print("find split test: ", find_split(y, x))
    tree = build_tree(y, x, 3)
    print(list(tree.keys()) )
    print(tree)
    xreg = np.linspace(0,1, 100)
    yreg = reg_tree(y, x, xreg, 3)
    plt.plot(xreg, yreg, marker = 'o')
    plt.show()
