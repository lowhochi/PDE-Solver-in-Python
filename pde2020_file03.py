# pde2020_file03.py
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import linalg as la
import sys
import pde2020_file03Fun
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
no_of_interval = 20 # divisible by 2**N
rMinMax = np.array([0.0, 1.0])
thetaMinMax = np.array([0.0, 1.0])
dr = (rMinMax[1] -rMinMax[0])/no_of_interval
dtheta = (thetaMinMax[1] -thetaMinMax[0])/no_of_interval
r = np.arange(rMinMax[0], rMinMax[1]+dr, dr)
theta = np.arange(thetaMinMax[0], thetaMinMax[1]+dtheta, dtheta)
R, Theta = np.meshgrid(r, theta) #[0,1]x[0,1]
size_of_R = R.shape[0]
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
N = int(2)
r_marked_point = np.arange(0.0, 1.0+1.0/2**N, 1.0/2**N)
r_marked_point[0] = dr # modification
theta_marked_point = np.arange(0.0, 1.0+1.0/2**N, 1.0/2**N)
Rmark, Thetamark = np.meshgrid(r_marked_point, theta_marked_point)

f = np.zeros(R.shape)
for p in range(0, size_of_R):
    for q in range(0, size_of_R):
        # f[p,q] = math.cos(2*math.pi*Theta[p,q]) + math.sin(2*math.pi*Theta[p,q])
        f[p,q] = -1.0
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
region = np.ndarray(shape=R.shape, dtype=object)
for p in range(0, size_of_R):  
    for q in range(1, size_of_R):
        rIndex, thetaIndex, s = pde2020_file03Fun.find_region(R[p,q], Theta[p,q], dr, N)
        region[p, q] = np.array([rIndex, thetaIndex, s])
for p in range(0, size_of_R):
    region[p, 0] = np.array([0, 0, 0])
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
triangle = np.ndarray(shape = (2**N, 2**N, 2), dtype=object)
for k in range(0, 2**N):
##    triangle[0, k, 0] = np.array([[dr, dr, 1.0/2**N], \
##                                     [float(k)/2**N, float(k+1)/2**N, float(k)/2**N]])
##    triangle[0, k, 1] = np.array([[dr, 1.0/2**N, 1.0/2**N], \
##                                     [float(k+1)/2**N, float(k+1)/2**N, float(k)/2**N]])
    for j in range(0, 2**N):
        triangle[j, k, 0] = np.array([[float(j)/2**N, float(j)/2**N, float(j+1)/2**N], \
                                         [float(k)/2**N, float(k+1)/2**N, float(k)/2**N]])
        triangle[j, k, 1] =  np.array([[float(j)/2**N, float(j+1)/2**N, float(j+1)/2**N], \
                                          [float(k+1)/2**N, float(k+1)/2**N, float(k)/2**N]])
        
adjRegion = pde2020_file03Fun.make_adjacent_region(R, Theta, Rmark, \
                                                   Thetamark, region, triangle, size_of_R, N)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
vMap, vGradient = pde2020_file03Fun.make_vMap(R, Theta, Rmark, Thetamark, \
                                   region, triangle, adjRegion, size_of_R, N)

Marray = pde2020_file03Fun.make_Marray(R, Theta, vGradient, size_of_R, dr, N)
farray = pde2020_file03Fun.make_farray(R, Theta, vMap, f, size_of_R, dr, N)

Uarray = la.solve(Marray, farray)
# checkSoln = np.allclose(np.dot(Marray.T, Uarray), farray)
u = np.zeros(R.shape)
Ntwo = int((2**N-1)*(2**N)+1)
for p in range(0,size_of_R):
    for q in range(0, size_of_R):
        temp = Uarray[0]*vMap[0,0][p,q]
        for row in range(1, Ntwo):
            jj = 1+ ((row-1)%(2**N-1))
            kk = (row-1)/(2**N-1)
            temp = temp + Uarray[row]*vMap[jj, kk][p,q]
        u[p,q] = temp
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# pde2020_file03Fun.graph_of_triangulation(Rmark, Thetamark, triangle, dr, N)
# pde2020_file03Fun.make_graph3d(vMap[1,1], R, Theta)
# pde2020_file03Fun.make_polar3d(u, R, Theta, size_of_R)
pde2020_file03Fun.graph_of_adjRegion(1, 1, r, theta, Rmark, Thetamark, \
                                    triangle, adjRegion, dr, N)
#pde2020_file03Fun.graph_of_triangulation(Rmark, Thetamark, triangle, dr, N)
plt.show()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
