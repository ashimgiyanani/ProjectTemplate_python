import numpy as np
import os

def dotprod(f,t):
# create an n x m array from 2 vectors of size n and m.
#    Resulting rows are the multiplication of each element of the first vector for all the elements of the second vector
#    f=np.array([2,4])
#    t=np.array([1,2,3,4,5])
#    k=funMatLabMultip(f,t)
#    print(k)
#    [[ 2  4  6  8 10]
#    [ 4  8 12 16 20]]


    if t.size==t.shape[0]:
        k=f[0]*t
        for i in f[1:]:
            j=i*t
            k=np.vstack((k,j))
    else:
        raise Exception('arrays should 1D arrays')
    return k


def dotdiv(f,t):
#   create an n x m array from 2 vectors of size n and m.
#    Resulting rows are the multiplication of each element of the first vector for all the elements of the second vector
#    f=np.array([2,4])
#    t=np.array([1,2,3,4,5])
#    k=funMatLabMultip(f,t)
#    print(k)
#    [[ 2  4  6  8 10]
#     [ 4  8 12 16 20]]

    if t.size==t.shape[0]:
        k=f[0]/t
        for i in f[1:]:
            j=i/t
            k=numpy.vstack((k,j))
    else:
        raise Exception('arrays should 1D arrays')
    return k

def printMatrix(a):
# print a matrix on the screen with integer values
   print("Matrix["+("%d" %a.shape[0])+"]["+("%d" %a.shape[1])+"]")
   rows = a.shape[0]
   cols = a.shape[1]
   for i in range(0,rows):
      for j in range(0,cols):
         print("%6.f" %a[i,j],end=' ')
      print(end='\n')
   print      


def printMatrixE(a):
   print("Matrix["+("%d" %a.shape[0])+"]["+("%d" %a.shape[1])+"]")
   rows = a.shape[0]
   cols = a.shape[1]
   for i in range(0,rows):
      for j in range(0,cols):
         print("%6.3f" %a[i,j],end=' ')
      print(end='\n')
   print      

# Examples:
# inf = float('inf')
# A = np.array( [[0,1.,4.,inf,3],
#      [1,0,2,inf,4],
#    [4,2,0,1,5],
#     [inf,inf,1,0,3],
#     [3,4,5,3,0]])

# printMatrix(A)    
# printMatrixE(A)    

##

# clear all variables in the middle of a Python script?
__saved_context__ = {}

def saveContext():
    import sys
    __saved_context__.update(sys.modules["__main__"].__dict__)

def restoreContext():
    import sys
    names = list(sys.modules["__main__"].__dict__.keys())
    for n in names:
        if n not in __saved_context__:
            del sys.modules["__main__"].__dict__[n]
    
    clear = restoreContext()

# saveContext()
# 
# hello = 'hi there'
# print hello             # prints "hi there" on stdout
# 
# restoreContext()
# 
# print hello             # throws an exception

def findi(lst, condition):
   return [int(i) for i, elem in enumerate(lst) if condition(elem)]
# find_indices(a, lambda e: e > 2)
# [2, 5]


def movingwindow(seq, size, step=1):
    # initialize iterators
    iters = [iter(seq) for i in range(size)]
    # stagger iterators (without yielding)
    [next(iters[i]) for j in range(size) for i in range(-1, -j-1, -1)]
    while(True):
        yield [next(i) for i in iters]
        # next line does nothing for step = 1 (skips iterations for step > 1)
        [next(i) for i in iters for j in range(step-1)]

def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor="")

    return ax


def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------
    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """
    if ax is None:
        ax = plt.gca()

    bootindex = sp.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        # Make coeffs of for polys
        pc = sp.polyfit(xs, ys + resamp_resid, 1)
        # Plot bootstrap cluster
        ax.plot(xs, sp.polyval(pc, xs), "b-", linewidth=2, alpha=3.0 / float(nboot))

    return ax

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()

    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax


# if "__main__" == __name__ :
# 	x = np.random.normal(size=100000)
# 	y = x * 3 + np.random.normal(size=100000)
# 	fig, ax = plt.subplots(figsize=(10,3))
# 	ax = density_scatter( x, y,ax=ax, fig=fig, bins = [30,30] )

# # Calculate the point density
# from scipy.stats import gaussian_kde
# x = Uhub[idx,0]
# y = TIh[idx,0]
# xy = np.vstack([x,y])
# z = gaussian_kde(xy)(xy)

# # Sort the points by density, so that the densest points are plotted last
# idx = z.argsort()
# x, y, z = x[idx], y[idx], z[idx]

# fig, ax = plt.subplots()
# ax.scatter(x, y, c=z, s=50)
# plt.show()
