from matplotlib.mlab import griddata
import numpy as np
# test case that scikits.delaunay fails on, but natgrid passes..
data = np.array([[-1, -1], [-1, 0], [-1, 1],
                 [ 0, -1], [ 0, 0], [ 0, 1],
                 [ 1, -1 - np.finfo(np.float_).eps], [ 1, 0], [ 1, 1],
                ])
x = data[:,0]
y = data[:,1]
z = x*np.exp(-x**2-y**2)
# define grid.
xi = np.linspace(-1.1,1.1,100)
yi = np.linspace(-1.1,1.1,100)
# grid the data.
zi = griddata(x,y,z,xi,yi)
