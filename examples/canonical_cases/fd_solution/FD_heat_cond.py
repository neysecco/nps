#Traditional solver for the 1-D Unstead Heat Transfer problem. 
'''
For this problem, we'll be using some concepts applied to solve the linear advection problem, 
but with second order central difference at the previous time.
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

if not os.path.exists('./heat_cond'):
    os.makedirs('./heat_cond')
    
#############################################################
## DOMAIN

#Conduction constant
k = 0.1
# Left edge temperature
Tl = 1.0
# Right edge temperature
Tr = 0.0
# Plate length
L = 1.0
# Simulation time
tf = 1.0
        
Fo = 0.50
# Generate refined boundaries
#nx
nx = 13
x = np.linspace(0,L,nx)
delta_x = x[1] - x[0]   # since it is a linear distribution 

#nt
delta_t = (Fo*delta_x**2)/k
nt = int(tf/delta_t)
t = np.linspace(0,tf,nt)

#mesh
X,T = np.meshgrid(x,t)

# It might be unnecessary to create a matrix for this problem. But it will be useful in the future.
Ugrid = np.zeros([nt,nx]) # Matrix that only receives the results from each time instance at each line.

## 1-D linear advection problem
'''
print("\nFo = " + str(k*delta_t/delta_x**2) + " || cond. for stability: Fo <= 1/4 \n")
'''

## Boundary condition

Ugrid[:,0] = Tl

## Analytical solution
def analytical_sol():
    # Initialize Output array
    Outputs = np.zeros([nt,nx])
    # Temperature difference
    deltaT = Tl - Tr
    # Select number of Fourier terms
    N = np.arange(1,501)
    # Fourier coefficients
    bn = L*deltaT/np.pi**2/N**2*(np.sin(np.pi*N) - np.pi*N)
    for ti in range(nt):
        for xi in range(nx):        
            Outputs[ti][xi] = deltaT*(1-x[xi]/L) + 2/L*np.sum(bn*np.sin(N*np.pi*x[xi]/L)*np.exp(-k*N**2*np.pi**2*t[ti]/L**2))
    return Outputs

Ugrid_ana = analytical_sol()

############################################################
## Execution
start_time = time.time()

for ti in range(1,nt): # time sweep, excluding the time at boundary where t == 0
    for xi in range(1,nx-1): # position sweep, excluding the position where x == -1
        Ugrid[ti,xi] = Ugrid[ti-1][xi] + Fo*(Ugrid[ti-1][xi+1] + Ugrid[ti-1][xi-1] - 2*Ugrid[ti-1][xi])

R_ana    = np.sum((Ugrid-Ugrid_ana)**2)
MSE_ana  = R_ana/(2*nx*nt) 

end_time = time.time()

print("\n\n## Solution for the 1-D Unsteady Heat Transfer problem")
print("Convergence time = " + str(end_time-start_time) + " seconds")
print("MSE = " + str(MSE_ana))
print(" ")
############################################################
## Plot the results
level_min=-0.05
level_max=1.05
# Contour plot
fig = plt.figure()
CS = plt.contourf(X, T, Ugrid, cmap=cm.viridis, levels=np.linspace(level_min,level_max,11), extent=[0, 1, 0, 1])
plt.xlabel(r'$x$',fontsize=18)
plt.ylabel(r'$t$',fontsize=18)
cb = plt.colorbar(CS)
cb.set_label(label=r'$u$',size=18)
cb.ax.tick_params(labelsize=14)
plt.tight_layout()
#plt.show()
fig.savefig("./heat_cond/FD_heat_cond_"+str(nx)+"_x_"+str(nt)+".pdf")

# Figure 2
fig2 = plt.figure(figsize=(8,8))
plt.subplot(3, 1, 1)
plt.plot(x, Ugrid_ana[0][:],'k',label='Analytical',linewidth=2)
plt.plot(x, Ugrid[0],'r',label='Traditional solution',linewidth=2)
plt.legend(loc='best',fontsize=18)
plt.title(r'$t=0$', fontsize=18) 
plt.ylabel(r'$u$',fontsize=18)
plt.xlabel(' ',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)

plt.subplot(3, 1, 2)
plt.plot(x, Ugrid_ana[int(len(t)/2)][:],'k',linewidth=2)
plt.plot(x, Ugrid_ana[int(len(t)/2)][:],'r',linewidth=2)
plt.title(r'$t='+str(t[int(len(t)/2)])+'$',fontsize=18)
plt.ylabel(r'$u$',fontsize=18)
plt.xlabel(' ',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)

plt.subplot(3, 1, 3)
plt.plot(x, Ugrid_ana[-1][:],'k',linewidth=2)
plt.plot(x, Ugrid[-1],'r',linewidth=2)
plt.title(r'$t=1$', fontsize=18)
plt.ylabel(r'$u$',fontsize=18)
plt.xlabel(r'$x$',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)

fig2.tight_layout()
fig2.savefig("./heat_cond/FD_heat_cond_"+str(nx)+"_x_"+str(nt)+"_slice.pdf")
#plt.show()