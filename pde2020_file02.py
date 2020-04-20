# pde2020_file02.py
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import linalg as la
import sympy
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm #colormap
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import  pde2020_file02Fun
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Dirichlet problem on [0,1]x[0,1]
# -u_xx - u_yy = f(x,y)
# u = 0 on boundary
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# basic variables on D = [0,1]x[0,1]
no_of_interval = 20 # (fixed)
xMinMax = np.array([0.0, 1.0])
yMinMax = np.array([0.0, 1.0])
dx = (xMinMax[1] - xMinMax[0])/no_of_interval
dy = dx
x = np.arange(xMinMax[0], xMinMax[1]+dx, dx)
y = np.arange(yMinMax[0], yMinMax[1]+dy, dy)
X, Y = np.meshgrid(x, y)
size_of_X = X.shape[0]
# finite element method on D
# no. of interior marked points = (2**N-1)**2
# no. of triangles in use = 2*(2**N)
# marked points = (i/2^N, j/2^N) w/. j = 0,1,2,...,2^N
N = int(3) 
x_marked_point = np.arange(0.0, 1.0+1.0/2**N, 1.0/2**N)
y_marked_point = x_marked_point # np.arange(0.0, 1.0+1.0/pow(2,N), 1.0/pow(2,N))
Xmark, Ymark = np.meshgrid(x_marked_point, y_marked_point)
# pde2020_file02Fun.graph_of_domain(Xmark, Ymark, N)

# define function f
# f = np.zeros(X.shape)
f, funString= pde2020_file02Fun.function_input(X, Y, size_of_X)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#Region = np.ndarray(shape=X.shape, dtype=object)
#triangleSet = np.ndarray(shape = (2**N, 2**N, 2), dtype=object)
Region, triangleSet = pde2020_file02Fun.make_region(X, Y, size_of_X, N)
# vMap = np.ndarray(shape=(2**N+1, 2**N+1), dtype=object)
# vNormal = np.ndarray(shape=(2**N+1, 2**N+1), dtype=object)
vMap, vGradient = pde2020_file02Fun.make_vMap(X, Y, Xmark, Ymark, \
                                            triangleSet, Region, size_of_X, N)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
Ntwo = (2**N -1)**2
# (jj, kk) refers to (2**N-1)*(kk -1) +(jj-1)
# vProduct[row, col] = grad(vJK) \cdot grad(vPQ) on D
# vProduct = np.ndarray(shape=[Ntwo, Ntwo], dtype=object)
# find M = [mij]'s
# Marray = np.zeros(shape=[Ntwo, Ntwo])
Marray = pde2020_file02Fun.make_Marray(X, vGradient, size_of_X, dx, N, Ntwo)
# fvProduct = np.ndarray(shape=[Ntwo], dtype=object)
# farray = np.zeros([Ntwo])
farray = pde2020_file02Fun.make_farray(X, vMap, f, size_of_X, dx, N, Ntwo)
           
Uarray = la.solve(Marray.T, farray)
# checkSoln = np.allclose(np.dot(Marray.T, Uarray), farray)
u = np.zeros(X.shape)
for p in range(0,size_of_X):
    for q in range(0, size_of_X):
        temp = 0
        for row in range(0, Ntwo):
            jj = 1 + (row%(2**N-1))
            kk = 1 + row/(2**N-1)
            temp = temp + Uarray[row]*vMap[jj, kk][p,q]
        u[p,q] = temp
print "max-value of u is: ", np.amax(u)
pde2020_file02Fun.make_graph3d(u, X, Y)
# pde2020_file02Fun.make_contour(u, X, Y)
plt.show()
# plt.close()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# pde2020_file02Fun.save_data(no_of_interval, xMinMax, yMinMax, x, y, size_of_X, \
#                             X, Y, f, u, N, Xmark, Ymark, funString)



