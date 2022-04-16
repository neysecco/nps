'''
This script solves the heat equation using the Gauss-Seidel method
and Finite Difference.
It is necessary to compile the Fortran code first with:
$ bash compile_heat_eq.sh
Then execute the code with
$ python3 FD_heat_eq.py

2147 x 4
'''

##########################################################
# IMPORTS
import numpy as np
import time
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from heat_eq import gauss_seidel

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'

if not os.path.exists('./heat_eq'):
    os.makedirs('./heat_eq')
    
##########################################################
# VARIABLES 
n = 301 # Number of nodes 
nx= n; ny= n 
MSE_stop = 7.40995337448186e-05
T = np.array(np.zeros((n,n)), order='F') # Temperature grid
use_fortran = True

# Correct fortran ordering if necessary
if use_fortran:
    T = np.array(T, order='F') # Temperature grid

##########################################################
def analytical_sol(Inputs):
    # Number of terms in the Fourier Series
    n_fourier = 150
    # Initializing grid to accumulate the terms of the Fourier series
    Outputs = np.zeros(np.shape(Inputs[0]))
    for  n in range(1,n_fourier+1):
        delta = 2.*((-1.)**n-1.)/n/np.pi/np.sinh(n*np.pi)*np.sin(n*np.pi*Inputs[0])*np.sinh(n*np.pi*(Inputs[1]-1.))
        Outputs = Outputs + delta #Increment of the grid
    return Outputs
        
###########################################################
# EXECUTION
start_time = time.time()
# Temperature solver
T = T*0
T[0,:] = 1 # Boundary condition
R = 0 # Start the residual with zero
MSE_R   = MSE_stop*2
MSE_ana = MSE_stop*2
x = np.linspace(0,1,n)
y = np.linspace(0,1,n)
#print('h:',x[1]-x[0])
Tgrid = np.meshgrid(x,y)
    
T_ana = analytical_sol(Tgrid)

while (MSE_ana>=MSE_stop):

    if not use_fortran:
        for i in range(1, n-1):
            for j in range(1,n-1):
                T[i,j] = (T[i-1,j] + T[i+1,j] + T[i,j-1] + T[i,j+1])/4 # Gauss
    else:
        gauss_seidel(T) # Fortran subroutine

    R_ana = np.sum((T-T_ana)**2)
    MSE_ana = R_ana/(2*n*n)

end_time = time.time()

x = np.linspace(0,1,n)
y = np.linspace(0,1,n)
Tgrid = np.meshgrid(x,y)
X,Y   = np.meshgrid(x,y) 

print("\n\n## Solution for the 2-D Steady Heat Transfer problem")
print("Convergence time = " + str(end_time-start_time) + " seconds")
print("MSE = " + str(MSE_ana))
print("nx = " + str(nx))
print("ny = " + str(ny))
print(" ")
#############################################################
level_min=-0.05
level_max=1.05
# Contour plot
fig = plt.figure()
CS = plt.contourf(X, Y, T_ana, cmap=cm.viridis, levels=np.linspace(level_min,level_max,11), extent=[0, 1, 0, 1])
plt.xlabel(r'$x$',fontsize=18)
plt.ylabel(r'$y$',fontsize=18)
cb = plt.colorbar(CS)
cb.set_label(label=r'$u$',size=18)
cb.ax.tick_params(labelsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
#plt.show()
fig.savefig("./heat_eq/FD_heat_eq_"+str(nx)+"_x_"+str(ny)+".pdf")


# Figure 2
T_ana = np.transpose(T_ana)
T     = np.transpose(T)
fig2  = plt.figure(figsize=(8,8))

plt.subplot(3, 1, 1)
plt.plot(y, T_ana[int(len(x)/5)][:],'k',label='Analytical',linewidth=2)
plt.plot(y, T[int(len(x)/5)][:],'r',label='FD solution',linewidth=2)
plt.legend(loc='best')
plt.title(r'$x='+str(x[int(len(x)/5)])+'$',fontsize=18)
plt.ylabel(r'$u$', fontsize=18)
plt.xlabel(' ',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)

plt.subplot(3, 1, 2)
plt.plot(y, T_ana[int(len(x)/2)][:],'k',linewidth=2)
plt.plot(y, T[int(len(x)/2)][:],'r',linewidth=2)
plt.title(r'$x='+str(x[int(len(x)/2)])+'$',fontsize=18)
plt.ylabel(r'$u$',fontsize=18)
plt.xlabel(' ',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)

plt.subplot(3, 1, 3)
plt.plot(y, T_ana[int(4*len(x)/5)][:],'k',linewidth=2)
plt.plot(y, T[int(4*len(x)/5)][:],'r',linewidth=2)
plt.title(r'$x='+str(x[int(4*len(x)/5)])+'$',fontsize=18)
plt.ylabel(r'$u$',fontsize=18)
plt.xlabel(r'$x$',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)

fig2.tight_layout()
fig2.savefig("./heat_eq/FD_heat_eq_"+str(nx)+"_x_"+str(ny)+"_slice.pdf")
#plt.show()
