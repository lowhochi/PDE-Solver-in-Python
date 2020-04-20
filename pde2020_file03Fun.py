# pde2020_file03Fun.py
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import linalg as la
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm #colormap
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def find_region(r0, theta0, dr, N): # dr<=r0<=1.0
    s = 1
    if (r0<1.0/2**N) and (r0>=dr):
        rIndex = 0
        thetaIndex = 2**N-1
        for k in range(0, 2**N):
            if (theta0>=float(k)/2**N) and (theta0<float(k+1)/2**N):
                thetaIndex = k
                break
        # line(0,j): theta = -(1./2**N)/(1./2**N-dr)*(r-dr) + theta[j] +1.0/2**N
        # j = 0, 1, ..., 2**N-1
        temp = 1.0/(dr*2**N-1.0)*(r0 -dr) + float(thetaIndex+1)/2**N
        if (theta0<temp):
            s = 0
        else:
            s = 1
    else:
        rIndex = 2**N-1
        thetaIndex = 2**N-1
        for j in range(1, 2**N):
            if (r0>=(float(j)/2**N)) and (r0<(float(j+1)/2**N)):
                rIndex = j
                break
        for k in range(0, 2**N):
            if (theta0>=(float(k)/2**N)) and (theta0<(float(k+1)/2**N)):
                thetaIndex = k
                break
        if ((r0+theta0)<float(rIndex +thetaIndex +1)/2**N):
            s = 0 # lower triangle of Rjk
        else:
            s = 1 # upper triangle of Rjk 
    return rIndex, thetaIndex, s

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def plane_function(v0, v1, v2, s, t):
    # each v1, v2, v3 is an np.array of size (2, 1)
    # v1 = np.array([x1, y1])
    # plane function f dtm by f(v0) = 1, f(v1) = f(v2) = 0
    # output: z = f(s,t)
    a1 = v2[1] -v1[1]
    a2 = v1[0] -v2[0]
    a3 = v1[0]*v2[1] -v2[0]*v1[1] +(v1[1] -v2[1])*v0[0]\
         +(v2[0] -v1[0])*v0[1]
    z = 1.0 -(float(a1)/a3)*(s -v0[0]) -(float(a2)/a3)*(t -v0[1])
    z = max(z, 0.0)
    gradVec = np.array([-float(a1)/a3, -float(a2)/a3])
    return z, gradVec
    # return z, nVec
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def graph_of_triangulation(Rmark, Thetamark, triangle, dr, N):                    
    for k in range(0, 2**N):
        for j in range(0, 2**N):
            for p in range(0,3):
                plt.plot([triangle[j,k,0][0, p], triangle[j,k,0][0, (p+1)%3]], \
                     [triangle[j,k,0][1, p], triangle[j,k,0][1, (p+1)%3]],\
                     linewidth = 2, linestyle='--', color = 'blue')   
    for j in range(0, 2**N):
        for k in range(1, 2**N):
            plt.scatter(Rmark[j,k], Thetamark[j,k], s=50, color='green')
    plt.plot([0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0], linewidth=2, color='blue')
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.grid(True)
    
def make_graph3d(uFun, X, Y):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X,Y,uFun, rstride=1, cstride=1, cmap=plt.get_cmap('hsv'), \
                          linewidth=0, antialiased=False)
    ax.set_zlim(-0.1, 1.1)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

def make_polar3d(uFun, R, Theta, size_of_R):
    X = np.zeros([size_of_R, size_of_R])
    Y = np.zeros([size_of_R, size_of_R])
    for p in range(0, size_of_R):
        for q in range(0, size_of_R):
            X[p, q] = R[p,q]*math.cos(2*math.pi*Theta[p,q])
            Y[p, q] = R[p,q]*math.sin(2*math.pi*Theta[p,q])
            # uFun0[p, q] = uFun[p, q]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X,Y,uFun, rstride=1, cstride=1, cmap=plt.get_cmap('hsv'), \
                          linewidth=0, antialiased=False)
    #plt.title(r'$-\Delta u = f$ on unit disk, $f = \cos(2\pi\theta) +\sin(2\pi\theta)$')
    ax.set_zlim(-1.1, 1.1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def make_adjacent_region(R, Theta, Rmark, Thetamark, region, triangle, \
                         size_of_R, N):
    adjRegion = np.ndarray(shape=(2**N, 2**N), dtype=object)
    # interior vertices
    # vJK w/. 1<=J,K<2**N
    for jj in range(1, 2**N):
        for kk in range(1, 2**N):
            vJK = np.array([Rmark[kk, jj], Thetamark[kk, jj]])
            adjRegion[jj,kk] = np.zeros([2**N, 2**N, 2])
            for j in range(0, 2**N):
                for k in range(0, 2**N):
                    lowerTriangleJK = triangle[j, k, 0]
                    upperTriangleJK = triangle[j, k, 1]
                    for col in range(0,3):
                        xL = lowerTriangleJK[0,col]
                        yL = lowerTriangleJK[1,col]
                        xU = upperTriangleJK[0,col]
                        yU = upperTriangleJK[1,col]    
                        if (xL==vJK[0]) and (yL==vJK[1]):
                            adjRegion[jj,kk][j , k, 0] = 1
                        if (xU==vJK[0]) and (yU==vJK[1]):
                            adjRegion[jj,kk][j, k, 1] = 1
    # vertices vJ0 (K=0)
    for jj in range(1, 2**N):
        adjRegion[jj,0] = np.zeros([2**N, 2**N, 2])
        # Region R_{0J} and R_{2**N-1,J}
        adjRegion[jj,0][jj-1, 0, 0] = 1
        adjRegion[jj,0][jj-1, 0, 1] = 1
        adjRegion[jj,0][jj, 0, 0] = 1
        adjRegion[jj,0][jj-1, 2**N-1, 1] = 1
        adjRegion[jj,0][jj, 2**N-1, 0] = 1
        adjRegion[jj,0][jj, 2**N-1, 1] = 1
    # vertices v0K (J=0)
    adjRegion[0,0] = np.zeros([2**N, 2**N, 2])
    adjRegion[0,0][0,0,0] = 1
    adjRegion[0,0][0, 2**N-1, 0] = 1
    adjRegion[0,0][0, 2**N-1, 1] = 1
    for kk in range(1,2**N):
        adjRegion[0, kk] = np.zeros([2**N, 2**N, 2])
        adjRegion[0,0][0, kk-1, 0] = 1
        adjRegion[0,0][0, kk-1, 1] = 1
        adjRegion[0,0][0, kk, 0] = 1
        adjRegion[0, kk][0, kk-1, 0] = 1
        adjRegion[0, kk][0, kk-1, 1] = 1
        adjRegion[0, kk][0, kk, 0] = 1 
    return adjRegion
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def graph_of_region(jj, kk, s, triangle):
    triangleJK = triangle[jj, kk, s]
    plt.plot([triangleJK[0,0], triangleJK[0,1], triangleJK[0,2], triangleJK[0,0]], \
             [triangleJK[1,0], triangleJK[1,1], triangleJK[1,2], triangleJK[1,0]], linewidth=2, color='blue')

def graph_of_adjRegion(jj, kk, r, theta, Rmark, Thetamark, triangle, adjRegion, dr, N):
    vJK = np.array([Rmark[kk,jj], Thetamark[kk,jj]])
    adjRegionJK = adjRegion[jj,kk]
    for p in range(0, 2**N):
        for q in range(0, 2**N):
            if adjRegionJK[p, q, 0]==1:
                graph_of_region(p, q, 0, triangle)
            if adjRegionJK[p, q, 1]==1:
                graph_of_region(p, q, 1, triangle)
    for k in range(0, 2**N):
        for j in range(0, 2**N):
            if k>0:
                plt.scatter(Rmark[j,k], Thetamark[j,k], s=50, color='red')
            for p in range(0,3):
                plt.plot([triangle[j,k,0][0, p], triangle[j,k,0][0, (p+1)%3]], \
                         [triangle[j,k,0][1, p], triangle[j,k,0][1, (p+1)%3]],\
                         linewidth = 1, linestyle='--', color = 'red')
    for p in range(1, len(r)-1):
        plt.plot([r[p], r[p]], [0,1], linewidth = 1, linestyle ='--', color = 'green')
        plt.plot([0,1], [theta[p],theta[p]], linewidth = 1, linestyle ='--', color = 'green')
    plt.scatter(vJK[0], vJK[1], s=50, color='red')
    plt.plot([0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0], linewidth=1, color='red')   
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.grid(True)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def make_vMap(R, Theta, Rmark, Thetamark, region, triangle, adjRegion, size_of_R, N):
    vMap = np.ndarray(shape=(2**N, 2**N), dtype=object)
    vGradient = np.ndarray(shape=(2**N, 2**N), dtype=object)
    for jj in range(1, 2**N):
        for kk in range(1, 2**N):
            vJK = np.array([Rmark[kk,jj], Thetamark[kk,jj]])
            vMap[jj, kk] = np.zeros(R.shape)
            vGradient[jj, kk] = np.zeros([size_of_R, size_of_R, 2])
            adjRegionJK = adjRegion[jj, kk]
            for p in range(0, size_of_R):
                for q in range(0, size_of_R):
                    regionPQ = region[p, q]
                    j0 = regionPQ[0]
                    k0 = regionPQ[1]
                    s0 = regionPQ[2]
                    if (adjRegionJK[j0, k0, s0]==1):
                        triangle0 = triangle[j0, k0, s0]
                        pos = 0
                        for t in range(0,3):
                            if (vJK[0]==triangle0[0, t]) and (vJK[1]==triangle0[1, t]):
                                pos = t
                                break
                        v2 = np.array([triangle0[0, pos-1], triangle0[1, pos-1]])
                        v3 = np.array([triangle0[0, (pos+1)%3], triangle0[1, (pos+1)%3]])
                        vtemp, gradTemp = plane_function(vJK, v2, v3, R[p,q], Theta[p,q])
                        vMap[jj, kk][p, q] = vtemp
                        vGradient[jj, kk][p, q, 0] = gradTemp[0]
                        vGradient[jj, kk][p, q, 1] = gradTemp[1]
    for jj in range(1, 2**N):
        vJ0 = np.array([Rmark[0, jj], Thetamark[0, jj]])
        vJtop = np.array([Rmark[2**N,jj], Thetamark[2**N,jj]])
        vMap[jj, 0] = np.zeros(R.shape)
        vGradient[jj, 0] = np.zeros([size_of_R, size_of_R, 2])
        adjRegionJ0 = adjRegion[jj, 0]
        for p in range(0, size_of_R):
            for q in range(0, size_of_R):
                regionPQ = region[p, q]
                j0 = regionPQ[0]
                k0 = regionPQ[1]
                s0 = regionPQ[2]
                if (adjRegionJ0[j0, k0, s0]==1):
                    triangle0 = triangle[j0, k0, s0]
                    pos = 0
                    for t in range(0,3):
                        if (vJ0[0]==triangle0[0,t]) and (vJ0[1]==triangle0[1,t]):
                            pos = t
                            break
                        if (vJtop[0]==triangle0[0,t]) and (vJtop[1]==triangle0[1,t]):
                            pos = t
                            break
                    v1 = np.array([triangle0[0, pos], triangle0[1, pos]])
                    v2 = np.array([triangle0[0, pos-1], triangle0[1, pos-1]])
                    v3 = np.array([triangle0[0, (pos+1)%3], triangle0[1, (pos+1)%3]])
                    vtemp, gradTemp = plane_function(v1, v2, v3, R[p,q], Theta[p,q])
                    vMap[jj, 0][p, q] = vtemp
                    vGradient[jj, 0][p, q, 0] = gradTemp[0]
                    vGradient[jj, 0][p, q, 1] = gradTemp[1]
    vMap[0,0] = np.zeros(R.shape)
    vGradient[0,0] = np.zeros([size_of_R, size_of_R, 2])
    for p in range(0, size_of_R):
        for q in range(0, size_of_R):
            vMap[0,0][p,q] =  max(1.0 - (2**N)*R[p,q], 0.0)
            if (R[p,q] < Rmark[1,1]):
                vGradient[0,0][p, q, 0] = -2**N
                vGradient[0,0][p, q, 1] = 0.0
    return vMap, vGradient
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def integral_on_disc(uFun, R, Theta, dr, size_of_R): #uFun = uFun(r, theta)
    part1 = 0
    for p in range(0, size_of_R-1):
        for q in range(1, size_of_R-1):
            part1 = part1 + uFun[p, q]*(2*math.pi*R[p,q]*dr**2)
    part2 = 0
    for p in range(0, size_of_R-1):
        part2 = part2 + uFun[p, size_of_R-1]*(math.pi*R[p,size_of_R-1]*dr**2)
    part3 = uFun[0,0]*(math.pi*dr**2/4.0)
    return part1 +part2 + part3
##uTest = np.zeros(R.shape)
##for p in range(0, size_of_R):
##    for q in range(0, size_of_R):
##        uTest[p, q] = 1.0
##test = pde2020_file03Fun.integral_on_disc(uTest, R, Theta, dr, size_of_R)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def make_Marray(R, Theta, vGradient, size_of_R, dr, N):
    Ntwo = int((2**N-1)*(2**N)+1)
    vProduct = np.ndarray(shape=[Ntwo, Ntwo], dtype=object)
    for row in range(1, Ntwo):
        for col in range(1, Ntwo):
            vProduct[row, col] = np.zeros(R.shape)
            # find m(vJK, vPQ)
            jj = 1+ ((row-1)%(2**N-1))
            kk = (row-1)/(2**N-1)
            pp = 1+ ((col-1)%(2**N-1))
            qq = (col-1)/(2**N-1)
            vGradientJK = vGradient[jj, kk]
            vGradientPQ = vGradient[pp, qq]
            for p in range(0, size_of_R):
                for q in range(1, size_of_R): # avoid R[p,q]=0.0
                    temp = vGradientJK[p, q, 0]*vGradientPQ[p, q, 0] \
                           + 1.0/(4*(math.pi**2)*(R[p, q]**2))*vGradientJK[p, q, 1]*vGradientPQ[p, q, 1]
                    vProduct[row, col][p, q] = temp
    vGradient00 = vGradient[0, 0]
    for col in range(1, Ntwo):
        vProduct[0, col] = np.zeros(R.shape)
        pp = 1+ ((col-1)%(2**N-1))
        qq = (col-1)/(2**N-1)
        vGradientPQ = vGradient[pp, qq]
        for p in range(0, size_of_R):
            for q in range(1, size_of_R): # avoid R[p,q]=0.0
                    temp = vGradient00[p, q, 0]*vGradientPQ[p, q, 0] \
                           + 1.0/(4*(math.pi**2)*(R[p, q]**2))*vGradient00[p, q, 1]*vGradientPQ[p, q, 1]
                    vProduct[0, col][p, q] = temp
    for row in range(1, Ntwo):
        vProduct[row, 0] = vProduct[0, row]
    vProduct[0,0] = np.zeros(R.shape)
    for p in range(0, size_of_R):
        for q in range(0, size_of_R):
            vProduct[0,0][p, q] = vGradient[0,0][p, q, 0]**2
    # 
    Marray = np.zeros(shape=[Ntwo, Ntwo])
    for row in range(0, Ntwo):
        for col in range(0, Ntwo):
            Marray[row, col] = integral_on_disc(vProduct[row, col], R, Theta, dr, size_of_R)
    return Marray
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def make_farray(R, Theta, vMap, f, size_of_R, dr, N):
    Ntwo = int((2**N-1)*(2**N)+1)
    fvProduct = np.ndarray(shape=[Ntwo], dtype=object)
    farray = np.zeros([Ntwo])
    for row in range(1, Ntwo):
        fvProduct[row] = np.zeros(R.shape)
        jj = 1+ ((row-1)%(2**N-1))
        kk = (row-1)/(2**N-1)
        for p in range(0, size_of_R):
            for q in range(0, size_of_R):
                fvProduct[row][p,q] = f[p,q]*(vMap[jj, kk][p,q])
        farray[row] =  integral_on_disc(fvProduct[row], R, Theta, dr, size_of_R)
    fvProduct[0] = np.zeros(R.shape)
    for p in range(0, size_of_R):
        for q in range(0, size_of_R):
            fvProduct[0][p,q] = f[p,q]*(vMap[0,0][p,q])
    farray[0] = integral_on_disc(fvProduct[0], R, Theta, dr, size_of_R)
    return farray
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

