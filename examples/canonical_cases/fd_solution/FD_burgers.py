#Traditional solver for the 1-D Burger's equation.
''' 
For this problem, we'll be solving the 1-D Burger's equation applying the Lax-Wendroff 2nd order scheme.
This scheme is also stated as a standard technique by Sofia (lectures notes) and Manshoor et. al. (article: Numerical Solution of Burger’s Equation
based on Lax-Friedrichs and Lax-Wendroff Schemes)
'''
import numpy as np
import time
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
from matplotlib.ticker import LinearLocator, FormatStrFormatter

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'

if not os.path.exists('./burgers'):
    os.makedirs('./burgers')
    
#############################################################
## DOMAIN AND BOUNDARIES

# Bounds
x0 = 0.0
xf = 1.0
t0 = 0.0
tf = 1.0

# Generate refined boundaries
nx = 21
nt = 21
nx_ana = 301
nt_ana = 301

MSE_ana = 10000
MSE_target = 3.341854081927567e-06

# Generate matrix with analytical solution at the validation points
def analytical_sol(nt,nx):
    # Analytical solution
    Outputs = np.zeros([nt,nx])

    x = np.linspace(x0,xf,nx_ana)
    t = np.linspace(t0,tf,nt_ana)

    for ti in range(nt):
        for xi in range(nx):
            # Compute break points
            xb1 = 0.3 + 0.1*t[ti]
            xb2 = 0.4 + 0.2*t[ti]
            xb3 = 0.6 + 0.2*t[ti]
            xb4 = 0.7 + 0.1*t[ti]
            if x[xi] <= xb1:
                Outputs[ti][xi] = 0.1
            elif x[xi] > xb1 and x[xi] <= xb2:
                Outputs[ti][xi] = 0.1 + 0.1*(x[xi]-xb1)/(xb2-xb1)
            elif x[xi] > xb2 and x[xi] <= xb3:
                Outputs[ti][xi] = 0.2
            elif x[xi] > xb3 and x[xi] <= xb4:
                Outputs[ti][xi] = 0.2 - 0.1*(x[xi]-xb3)/(xb4-xb3)
            elif x[xi] > xb4:
                Outputs[ti][xi] = 0.1
    return Outputs

Ugrid_ana = analytical_sol(nt_ana, nx_ana)
x_ana = np.linspace(x0,xf,nx_ana)
t_ana = np.linspace(t0,tf,nt_ana)
X_ana,T_ana = np.meshgrid(x_ana,t_ana)

while MSE_ana > MSE_target:
    x = np.linspace(x0,xf,nx)
    t = np.linspace(t0,tf,nt)
    X,T= np.meshgrid(x,t) 

    # Save the results to this grid
    Ugrid = np.zeros([nt,nx]) # Matrix that only receives the results from each time instance at each line.

    ## 1-D linear advection problem
    delta_x = x[1] - x[0]   # since it is a linear distribution 
    delta_t = t[1] - t[0]   # From the original inputs, it gives a*delta_t/delta_x == 0.5, which matches the stability conditions

    '''
    print('delta_x = ', delta_x)
    print('delta_t = ', delta_t)
    print(' ')
    '''
    # Initial condition
    Ugrid[:,0] = 0.1
    Ugrid[:,-1] = 0.1
    for xi in range(nx):
        # Compute break points
        xb1 = 0.3
        xb2 = 0.4
        xb3 = 0.6
        xb4 = 0.7
        if x[xi] <= xb1:
            Ugrid[0][xi] = 0.1
        elif x[xi] > xb1 and x[xi] <= xb2:
            Ugrid[0][xi] = 0.1 + 0.1*(x[xi]-xb1)/(xb2-xb1)
        elif x[xi] > xb2 and x[xi] <= xb3:
            Ugrid[0][xi] = 0.2
        elif x[xi] > xb3 and x[xi] <= xb4:
            Ugrid[0][xi] = 0.2 - 0.1*(x[xi]-xb3)/(xb4-xb3)
        elif x[xi] > xb4:
            Ugrid[0][xi] = 0.1

    ############################################################
    ## Execution
    start_time = time.time()

    for ti in range(1,nt): # time sweep, excluding the time at boundary where t == 0
        for xi in range(1,nx-1): # position sweep, excluding the position where x == -1
            Ugrid_p = 0.5*(Ugrid[ti-1][xi+1] + Ugrid[ti-1][xi]) - (delta_t/(4*delta_x))*(Ugrid[ti-1][xi+1]**2 - Ugrid[ti-1][xi]**2)
            Ugrid_m = 0.5*(Ugrid[ti-1][xi] + Ugrid[ti-1][xi-1]) - (delta_t/(4*delta_x))*(Ugrid[ti-1][xi]**2 - Ugrid[ti-1][xi-1]**2)
            Ugrid[ti,xi] = Ugrid[ti-1][xi] - (delta_t/(2*delta_x))*(Ugrid_p**2 - Ugrid_m**2)

    end_time = time.time()

    # Interpolate solution to the validation points
    from scipy.interpolate import RectBivariateSpline as spline
    Ugrid_int = spline(t,x,Ugrid).ev(T_ana[:],X_ana[:]).reshape(X_ana.shape)

    R_ana    = np.sum((Ugrid_int-Ugrid_ana)**2)
    MSE_ana  = R_ana/(2*nx_ana*nt_ana) 

    nx = nx+10
    nt = nt+10

nx = nx-10
nt = nt-10

print("\n\n## Solution for the 1-D Burgers' equation problem")
print("Convergence time = " + str(end_time-start_time) + " seconds")
print("MSE = " + str(MSE_ana))
print("nx = " + str(nx))
print("nt = " + str(nt))
print(" ")

############################################################
## Plot the results
level_min=0.095
level_max=0.21

# Contour plot
fig = plt.figure()
CS = plt.contourf(X, T, Ugrid, cmap=cm.viridis, levels=np.linspace(level_min,level_max,11), extent=[0, 1, 0, 1])
plt.xlabel(r'$x$',fontsize=18)
plt.ylabel(r'$t$',fontsize=18)
cb = plt.colorbar(CS)
cb.set_label(label=r'$u$',size=18)
plt.tight_layout()
#plt.show()
fig.savefig("./burgers/FD_burgers_"+str(nx)+"_x_"+str(nt)+".pdf")

# Figure 2
fig2 = plt.figure(figsize=(8,6))
plt.subplot(2, 1, 1)
plt.plot(x_ana, Ugrid_ana[0][:],'k',label='Analytical',linewidth=2)
plt.plot(x, Ugrid[0],'r',label='FD solution',linewidth=2)
plt.legend(loc='best',fontsize=18)
plt.title(r'$t=0$', fontsize=18)
plt.ylabel(r'$u$',fontsize=18)
plt.xlabel(' ',fontsize=14)
plt.grid(False)

plt.subplot(2, 1, 2)
plt.plot(x_ana, Ugrid_ana[-1][:],'k',linewidth=2)
plt.plot(x, Ugrid[-1],'r',linewidth=2)
plt.title(r'$t=1$', fontsize=18)
plt.ylabel(r'$u$',fontsize=18)
plt.xlabel(r'$x$',fontsize=18)
plt.grid(False)

fig2.tight_layout()
fig2.savefig("./burgers/FD_burgers_"+str(nx)+"_x_"+str(nt)+"_slice.pdf")
#plt.show()