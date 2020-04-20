# finite_element_method01.py
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import linalg as la
import sys
# import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm #colormap
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# example1
# u_t = u_xx, u(x,0) = exp(-x**2)
def example1():
    # start_time = 0
    end_time = 1.0
    h = 0.001 # dt = h
    no_of_step = int(end_time/h)
   
    max_xValue = 10.0
    min_xValue = -10.0
    no_of_xInterval = 200
    dx = (max_xValue-min_xValue)/no_of_xInterval
    
    # delta_x = (max_xValue -min_xValue)/no_of_xInterval
    t = h*np.array(range(no_of_step+1))
    x = np.zeros(no_of_xInterval +1)
    u = np.zeros([no_of_xInterval+1, no_of_step+1])
    # u[j,n] = value of u(x[j], t[n])
    for j in range(0, no_of_xInterval +1):
        x[j] = min_xValue + j*(max_xValue -min_xValue)/no_of_xInterval
        if (j!=0) and (j!=no_of_xInterval): 
            u[j,0] = math.exp(-x[j]**2)
    # finite difference method
    for n in range(0, no_of_step):
        for j in range(1, no_of_xInterval):
            u[j,n+1] = u[j,n] + h/(dx**2)*(u[j+1,n] -2*u[j,n] +u[j-1,n])
    #ax = plt.gca()
    #ax.plot(x, u[:, 0], color='b')
    print "Number of step = ", no_of_step
    print "h/(dx**2) = ", float(h/dx**2)
    time_interval = end_time/5
    colorSet = ['blue', 'green', 'magenta', 'cyan', 'purple', 'red']
    color_count = 0
    for n in range(0,no_of_step+1):
        current_time = t[n]
        if (current_time in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
            # print 'current_time =', current_time 
            # print 'plot number = ', color_count
            myString = 't = ' + str(t[n])
            plt.plot(x, u[:,n], color= colorSet[color_count], linewidth=1.5)
            plt.text(0.0, u[no_of_xInterval/2 ,n], myString, fontsize=14)
            color_count = color_count +1
    #ax.plot(x, u[:, no_of_step], color='r')
    plt.axis([-10.0, 10.0, 0.0,1.10])
    plt.xlabel('x', fontsize = 20)
    plt.ylabel('u', fontsize = 20)
    plt.xticks(np.arange(-10.0, 11.0, 2.0))
    plt.yticks(np.arange(0.0, 1.01, 0.1))
    plt.grid(True)
    plt.title(r'$u_t = u_{xx}$, $u(x,0)=e^{-x^2}$', fontsize = 20)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.show()
    #plt.savefig('example1.png')
    #plt.close()

# example2
# u_xx + u_yy = 0
# square boundary [0,1]x[0,1]
def example2():
    step = 100 # no of steps in iteration
    xMin = 0.0
    xMax = 1.0
    yMin = 0.0
    yMax = 1.0
    no_of_interval = 10
    dx = (xMax - xMin)/no_of_interval
    dy = dx
    x = np.arange(xMin, xMax +dx, dx)
    y = np.arange(yMin, yMax +dy, dy)
    u = np.zeros([no_of_interval+1, no_of_interval+1],dtype=float)
    uNew = np.zeros([no_of_interval+1, no_of_interval+1],dtype=float)
    # implement boundary condition
    u[0, 0] = 0
    u[0, no_of_interval] = -1.0
    u[no_of_interval, 0] = 1.0
    u[no_of_interval, no_of_interval] = 0
    for j in range(1, no_of_interval):
        u[j, 0] = x[j]**2
        u[j, no_of_interval] = x[j]**2-1.0
        u[0, j] = -y[j]**2
        u[no_of_interval, j] = 1.0-y[j]**2
    #for j in range(0, no_of_interval+1):    
        # u[j, 0] = 324*(x[j]**2)*(1-x[j])
        # u[j, 1] = 0
    # for k in range(0,no_of_interval+1):
        # u[0,k] = 0
        # u[1,k] = 0
    # first guess
    for j in range(1, no_of_interval):
        for k in range(1, no_of_interval):
            u[j,k] = float(no_of_interval-k)/(no_of_interval)*u[j,0] \
                     + float(k)/(no_of_interval)*u[j,no_of_interval] \
                     + float(no_of_interval-j)/(no_of_interval)*u[0,k] \
                     + float(j)/(no_of_interval)*u[no_of_interval,k] 
    # Jacobi iteration
    for n in range(1, step+1):
        # print 'current step = ', n
        for j in range(1,no_of_interval):
            for k in range(1,no_of_interval):
                uNew[j,k] = (1.0/4)*(u[j+1,k] +u[j-1,k] + u[j,k+1] +u[j,k-1])
        # update u by uNew
        for j in range(1,no_of_interval):
            for k in range(1,no_of_interval):
                u[j,k]= uNew[j,k]
    return x, y, u

def graph_of_example2():
    x, y, u = example2()
    # np.amax(u) = 47.985696
    # contour map of u(x,y)
    X, Y = np.meshgrid(x, y)
    # U = np.zeros([no_of_interval+1, no_of_interval+1], dtype=float)
    # marked_levels = np.arange(0.0, 52.0, 2.0)
    #marked_levels = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 20.0, 24.0, \
    #                 28.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0]
    CS = plt.contourf(X, Y, u.T, extend='both')
    #CS = plt.contourf(X, Y, u.T, levels = marked_levels, extend='both') # cmap = 'RdGy'
##    CS.cmap.set_under('black')
    #plt.clabel(CS, fontsize=10, inline=True)
    plt.colorbar()
##    plt.title(r'$u_{xx} + u_{yy} = 0$', fontsize = 20)
##    myString = r'$u(x,0) = 324x^2(1-x)$'
##    plt.text(0.3, -0.05, myString, fontsize=14)
##    plt.text(0.4, 1.02, r'$u(x,1) = 0$', fontsize=14)
##    plt.text(-0.05, 0.5, r'$u(0,y)=0$', fontsize=14, rotation='vertical')
##    plt.text(1.02, 0.5, r'$u(1,y)=0$', fontsize=14, rotation='vertical')
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.xlabel('x', fontsize = 20)
    plt.ylabel('y', fontsize = 20)
    plt.xticks(np.arange(-0.1, 1.01, 0.1), fontsize=14)
    plt.yticks(np.arange(-0.1, 1.01, 0.1), fontsize=14)
    plt.grid(True)
    plt.show()
    #plt.savefig('example2.png')

def graph3d_of_example2():
    x, y, u = example2()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x,y)
    surf = ax.plot_surface(X,Y,u.T, rstride=1, cstride=1, cmap=plt.get_cmap('PuOr'), \
                           linewidth=0, antialiased=False)
    ax.view_init(elev=10, azim=300)
    plt.show()

# # # # # # # # # # # # # # # # # # # # MAIN # # # # # # # # # # # # # # # # # # # # 
# example1()
# u= example2()
graph_of_example2()
#graph3d_of_example2()

    
    
