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

def function_input(X, Y, size_of_X):
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    funString = str(raw_input('input function f(x,y)\n'))
    finput = sympy.sympify(funString)
    # finput = 2*(x+y-x**2-y**2)
    f = np.zeros(X.shape)
    for p in range(0, size_of_X):
        for q in range(0, size_of_X):
            f[p, q] = finput.subs(x, X[p,q]).subs(y, Y[p,q])
            # f[p, q] = 2*(X[p,q] +Y[p,q] - X[p,q]**2 -Y[p,q]**2)
    return f, funString

def graph_of_domain(Xmark, Ymark, N):
    #Xmark = np.reshape(Xmark, (1,int((2**N+1)**2)) )
    #Ymark = np.reshape(Ymark, (1,int((2**N+1)**2)) )
    plt.figure()
    plt.scatter(Xmark, Ymark, s=50, color='green')
    for j in range(0,2**N+1):
        plt.plot(Xmark[j,:], Ymark[j,:], linewidth=2, color='blue')
        plt.plot(Xmark[:,j], Ymark[:,j], linewidth=2, color='blue')
    for j in range(1,2**N+1):
        plt.plot([Xmark[j,0], Xmark[0,j]], [Ymark[j,0], Ymark[0,j]], linewidth=2, color='blue')
        plt.plot([Xmark[j,2**N], Xmark[2**N,j]], [Ymark[j,2**N], Ymark[2**N,j]], linewidth=2, color='blue')
    plt.plot([0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0], linewidth=2, color='blue')
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def find_region(xvalue, yvalue, N):
    xregion= 2**N-1
    yregion = 2**N-1
    s = 1
    for j in range(0, 2**N):
         for k in range(0, 2**N):
            if (xvalue>=(float(j)/2**N)) and (xvalue<(float(j+1)/2**N)):
                xregion = j
            if (yvalue>=(float(k)/2**N)) and (yvalue<(float(k+1)/2**N)):
                yregion = k
    if (xvalue +yvalue)<(float(xregion)/2**N + float(yregion)/2**N +1.0/2**N):
        s = 0 # lower triangle of Rjk
    else:
        s = 1 # upper triangle of Rjk        
    return xregion, yregion, s

# find the region index of (x,y) on D
def make_region(X, Y, size_of_X, N):
    # Region[p, q] = [j, k, s]
    # [j, k]: (X[p,q], Y[p,q]) belongs to the rectangle Rjk
    # s=0: (X[p,q], Y[p,q]) is in the lower triangle
    # s=1: (X[p,q], Y[p,q]) is in the higher triangle
    Region = np.ndarray(shape=X.shape, dtype=object)
    for p in range(0, size_of_X):
       for q in range(0, size_of_X):
           xregion, yregion, s = find_region(X[p,q], Y[p,q], N)
           Region[p, q] = np.array([xregion, yregion, s])
    # triangleSet = set of vertices of triangular regions
    triangleSet = np.ndarray(shape = (2**N, 2**N, 2), dtype=object)
    for j in range(0,2**N):
        for k in range(0,2**N):
            triangleSet[j,k,0] = np.array([[float(j)/2**N, float(j)/2**N, float(j+1)/2**N], \
                                           [float(k)/2**N, float(k+1)/2**N, float(k)/2**N]])
            triangleSet[j,k,1] =  np.array([[float(j)/2**N, float(j+1)/2**N, float(j+1)/2**N], \
                                            [float(k+1)/2**N, float(k+1)/2**N, float(k)/2**N]])
    return Region, triangleSet
    
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
    nVec = np.array([-float(a1)/a3, -float(a2)/a3])
    return z, nVec

def integration_on_D(g, size_of_X, dx):
    part1 = (g[0, 0] + g[0, size_of_X-1] + g[size_of_X-1, 0] +g[size_of_X-1, size_of_X-1])\
            *(dx**2/4.0)
    part2 = 0
    part3 = 0
    for p in range(1, size_of_X-1):
        part2 = part2 + g[size_of_X-1, p] +g[0, p] +g[p, 0] +g[p, size_of_X-1]
    part2 = part2*(dx**2/2.0)
    for p in range(1,size_of_X-1):
        for q in range(1,size_of_X-1):
            part3 = part3 + g[p,q]*(dx**2)
    return (part1 +part2 +part3)

def make_vMap(X, Y, Xmark, Ymark, triangleSet, Region, size_of_X, N):
    vMap = np.ndarray(shape=(2**N+1, 2**N+1), dtype=object)
    vGradient = np.ndarray(shape=(2**N+1, 2**N+1), dtype=object)
    # vJK = 0 if J or K = 0, 2**N
    for jj in range(0, 2**N+1):
        vMap[0, jj] = np.zeros(X.shape)
        vMap[2**N, jj] = np.zeros(X.shape)
        vMap[jj, 0] = np.zeros(X.shape)
        vMap[jj, 2**N] = np.zeros(X.shape)
        vGradient[0, jj] = np.zeros([size_of_X, size_of_X, 2])
        vGradient[2**N, jj] = np.zeros([size_of_X, size_of_X, 2])
        vGradient[jj, 0] =  np.zeros([size_of_X, size_of_X, 2])
        vGradient[jj, 2**N] = np.zeros([size_of_X, size_of_X, 2])
    # vertex vJK
    # jj, kk = 1, ... , 2**N-1
    for jj in range(1, 2**N):
        for kk in range(1, 2**N):
            vJK = np.array([Xmark[jj,kk], Ymark[jj,kk]])
            vMap[jj,kk] = np.zeros(X.shape)
            vGradient[jj,kk] = np.zeros([size_of_X, size_of_X, 2])
            inRegion = np.zeros([2**N, 2**N, 2])
            for j in range(0, 2**N):
                for k in range(0, 2**N):
                    lowerTriangleJK = triangleSet[j, k, 0]
                    upperTriangleJK = triangleSet[j, k, 1]
                    for col in range(0,3):
                        xL = lowerTriangleJK[0,col]
                        yL = lowerTriangleJK[1,col]
                        xU = upperTriangleJK[0,col]
                        yU = upperTriangleJK[1,col]    
                        if (xL==vJK[0]) and (yL==vJK[1]):
                            inRegion[j , k, 0] = 1
                        if (xU==vJK[0]) and (yU==vJK[1]):
                            inRegion[j, k, 1] = 1
            for p in range(0, size_of_X):
                for q in range(0, size_of_X): 
                # (X[p,q], Y[p,q]) belongs to Rjk, s=lower/upper
                    regionPQ = Region[p,q] 
                    j0 = regionPQ[0]
                    k0 = regionPQ[1]
                    s0 = regionPQ[2]
                    if inRegion[j0, k0, s0]==1:
                        vertexSet = triangleSet[j0, k0, s0]
                        pos = 0
                        for k in range(0,3):
                            if (vJK[0]==vertexSet[0,k]) and (vJK[1]==vertexSet[1,k]):
                                pos = k
                        v2 = np.array([vertexSet[0, (pos-1)], vertexSet[1, (pos-1)]])
                        v3 = np.array([vertexSet[0, (pos+1)%3], vertexSet[1, (pos+1)%3]])
                        vValueTemp, nVecTemp =  plane_function(vJK, v2, v3, X[p,q], Y[p,q])
                        vMap[jj, kk][p, q] = vValueTemp
                        vGradient[jj, kk][p,q,0] = nVecTemp[0]
                        vGradient[jj, kk][p,q,1] = nVecTemp[1]                 
    return vMap, vGradient

def make_Marray(X, vGradient, size_of_X, dx, N, Ntwo):
    # Ntwo = (2**N -1)**2
    # (jj, kk) refers to (2**N-1)*(kk -1) +(jj-1)
    # vProduct[row, col] = grad(vJK) \cdot grad(vPQ) on D
    vProduct = np.ndarray(shape=[Ntwo, Ntwo], dtype=object)
    for row in range(0, Ntwo):
        for col in range(0, Ntwo):
            vProduct[row, col] = np.zeros(X.shape)
            jj = 1 + (row%(2**N-1))
            kk = 1 + row/(2**N-1)
            pp = 1 + (col%(2**N-1))
            qq = 1 + col/(2**N-1)
            vGradientJK = vGradient[jj, kk]
            vGradientPQ = vGradient[pp, qq]
            for p in range(0, size_of_X):
                for q in range(0, size_of_X):
                    temp =  vGradientJK[p, q, 0]*vGradientPQ[p, q, 0]\
                        + vGradientJK[p, q, 1]*vGradientPQ[p, q,1]
                    vProduct[row, col][p, q] = temp
    # find M = [mij]'s
    Marray = np.zeros(shape=[Ntwo, Ntwo])
    for row in range(0, Ntwo):
        for col in range(0, Ntwo):
            Marray[row, col] = integration_on_D(vProduct[row, col], size_of_X, dx)

    return Marray

def make_farray(X, vMap, f, size_of_X, dx, N, Ntwo):
    #Ntwo = (2**N -1)**2
    fvProduct = np.ndarray(shape=[Ntwo], dtype=object)
    farray = np.zeros([Ntwo])
    for row in range(0, Ntwo):
        fvProduct[row] = np.zeros(X.shape)
        jj = 1 + (row%(2**N-1))
        kk = 1 + row/(2**N-1)
        for p in range(0, size_of_X):
            for q in range(0, size_of_X):
                fvProduct[row][p,q] = f[p,q]*(vMap[jj, kk][p,q])
        farray[row] =  integration_on_D(fvProduct[row], size_of_X, dx)
    return farray

def make_graph3d(uFun, X, Y):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X,Y,uFun, rstride=1, cstride=1, cmap=plt.get_cmap('hsv'), \
                          linewidth=0, antialiased=False)
    plt.title(r'$-u_{xx} -u_{yy} = f$, $u = 0$ on $\partial D$', fontsize=20)
    ax.view_init(elev=10, azim=300)
    # fig.savefig('example3.png')
    
def make_contour(uFun, X, Y):
    marked_levels = np.arange(0.01, 0.1, 0.01)
    CS = plt.contour(X, Y, uFun, levels = marked_levels, colors='k')
    plt.clabel(CS, inline=True, fontsize=10)
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.grid(True)

def create_savefile():
    f = open("pde2020_file02save.txt", "w+")
    f.write("file name: pde2020_file02save.txt")
    f.close()

##pde2020_file02Fun.save_data(no_of_interval, xMinMax, yMinMax, x, y, size_of_X, \
##                            X, Y, f, u N, Xmark, Ymark)
def save_data(no_of_interval, xMinMax, yMinMax, x, y, size_of_X, \
              X, Y, f, u, N, Xmark, Ymark, funString):
    dataNo = int(raw_input('Set the dataNo:')) #1001, 1002, so on
    g = open("pde2020_file02save.txt", "a")
    g.write("\ndataNo%d" %dataNo)
    g.write("\nfunString%s" %funString)
    # basic parameters
    g.write("\nno_of_interval"+str(no_of_interval))
    g.write("\nxmin"+str(xMinMax[0]))
    g.write("\nxmax"+str(xMinMax[1]))
    g.write("\nymin"+str(yMinMax[0]))
    g.write("\nymax"+str(yMinMax[1]))
    g.write("\nxarray")
    for j in range(0, no_of_interval+1):
        g.write(str(x[j])+" ")
    g.write("\nyarray")
    for j in range(0, no_of_interval+1):
        g.write(str(y[j])+" ")
    g.write("\nsize_of_X"+str(size_of_X))
    g.write("\nXmatrix")
    for p in range(0,size_of_X):
        g.write("\n")
        for q in range(0,size_of_X):
            g.write(str(X[p,q])+" ")
    g.write("\nYmatrix")
    for p in range(0,size_of_X):
        g.write("\n")
        for q in range(0,size_of_X):
            g.write(str(Y[p,q])+" ")
    g.write("\nfmatrix")
    for p in range(0,size_of_X):
        g.write("\n")
        for q in range(0,size_of_X):
            g.write(str(f[p,q])+" ")
    g.write("\nUmatrix")
    for p in range(0,size_of_X):
        g.write("\n")
        for q in range(0,size_of_X):
            g.write(str(u[p,q])+" ")
    g.write("\nN"+str(N))
    g.write("\nXmark")
    for jj in range(0, 2**N+1):
        g.write("\n")
        for kk in range(0, 2**N+1):
            g.write(str(Xmark[jj,kk])+" ")
    g.write("\nYmark")
    for jj in range(0, 2**N+1):
        g.write("\n")
        for kk in range(0, 2**N+1):
            g.write(str(Ymark[jj,kk])+" ")
    g.write("\n")
    g.close()

def read_data(dataNo):
    g = open("pde2020_file02save.txt", "r")
    gread = g.readlines()
    g.close()
    key = 'dataNo'+str(dataNo)
    keynum = 0
    for num, line in enumerate(gread, 0):
        line = line.rstrip()
        if (line==key):
            keynum = num
    print 'data starts from line', str(keynum)
    funString = gread[keynum+1][9:-1]
    no_of_interval = int(gread[keynum+2][14:-1])
    xmin = float(gread[keynum+3][4:-1])
    xmax = float(gread[keynum+4][4:-1])
    ymin = float(gread[keynum+5][4:-1])
    ymax = float(gread[keynum+6][4:-1])
    # # # # #
    xString = gread[keynum+7][6:-1]
    xString = (xString.split(' '))[0:no_of_interval+1]
    xList = [float(num) for num in xString]
    xarray = np.array(xList)
    # # # # #
    yString = gread[keynum+8][6:-1]
    yString = (yString.split(' '))[0:no_of_interval+1]
    yList = [float(num) for num in yString] 
    yarray = np.array(yList)
    # # # # # Xmatrix
    size_of_X =  int(gread[keynum+9][9:-1])
    Xmatrix = np.zeros([size_of_X, size_of_X])
    Ymatrix = np.zeros([size_of_X, size_of_X])
    f = np.zeros([size_of_X, size_of_X])
    u = np.zeros([size_of_X, size_of_X])
    for j in range(0, size_of_X):
        xrowString = gread[keynum+11+j][:-1]
        yrowString = gread[keynum+size_of_X+12+j][:-1]
        frowString = gread[keynum+2*size_of_X+13+j][:-1]
        urowString = gread[keynum+3*size_of_X+14+j][:-1]
        xrowString = (xrowString.split(' '))[0: size_of_X]
        yrowString = (yrowString.split(' '))[0: size_of_X]
        frowString = (frowString.split(' '))[0: size_of_X]
        urowString = (urowString.split(' '))[0: size_of_X]
        for k in range(0, size_of_X):
            Xmatrix[j, k] = float(xrowString[k])
            Ymatrix[j, k] = float(yrowString[k])
            f[j, k] = float(frowString[k])
            u[j, k] = float(urowString[k])
    N = int(gread[keynum+ 4*size_of_X +14][1:-1])
    Xmark = np.zeros([2**N+1, 2**N+1])
    Ymark = np.zeros([2**N+1, 2**N+1])
    for j in range(0, 2**N+1):
        XmarkrowString = gread[keynum+4*size_of_X+16+j][:-1]
        XmarkrowString = (XmarkrowString.split(' '))[0:2**N+1]
        YmarkrowString = gread[keynum+4*size_of_X+2**N+18+j][:-1]
        YmarkrowString = (YmarkrowString.split(' '))[0:2**N+1]
        for k in range(0, 2**N+1):
            Xmark[j,k] = float(XmarkrowString[k])
            Ymark[j,k] = float(YmarkrowString[k])
    dataDict = {"no_of_interval": no_of_interval, \
                "xarray": xarray, \
                "yarray": yarray, \
                "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, \
                "size_of_X": size_of_X, "Xmatrix": Xmatrix, \
                "Ymatrix": Ymatrix, "f": f, "u": u, \
                "N": N, "Xmark": Xmark, "Ymark": Ymark, \
                "funString": funString }
    return dataDict     
# dataDict = read_data(1001)
# dataDict["no_of_interval"], dataDict["xarray"], dataDict["yarray"], etc




