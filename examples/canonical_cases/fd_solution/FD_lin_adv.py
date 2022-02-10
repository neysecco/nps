#Traditional solver for the 1-D Linear Advection problem. 
'''
For this problem, we'll be using the first-order Gudonov Method, 
which applies a backward finite difference for the x derivative (for stability reasons) 
and foward difference for the time derivative (since we want to use information from the previous time level). 

We got these equations from the "International Centre for Theoretical Physics" lecture notes.
You can find more information from the documents provided at the google drive, or directly from the following link:
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi0zZLz6ePxAhWnKrkGHWDsBxgQFnoECAkQAA&url=http%3A%2F%2Findico.ictp.it%2Fevent%2Fa06220%2Fsession%2F18%2Fcontribution%2F10%2Fmaterial%2F0%2F2.pdf&usg=AOvVaw2REuXN08QCfnXwA_bMH6O7
http://indico.ictp.it/
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

if not os.path.exists('./lin_adv'):
    os.makedirs('./lin_adv')
    
find_opt = False
    
#############################################################
## DOMAIN AND BOUNDARIES

# Bounds
x0 = -1.0
xf = 1.0
t0 = 0.0
tf = 1.0

# Step position
xs1 = -0.75
xs2 = -0.25

# Generate refined boundaries
nx = 4711# 301
nt = 4711# 301

MSE_ana    = 100000
MSE_target = 0.0011362367862456778

while MSE_ana > MSE_target:
    x = np.linspace(x0,xf,nx)
    t = np.linspace(t0,tf,nt)

    X,T = np.meshgrid(x,t)

    # It might be unnecessary to create a matrix for this problem. But it will be useful in the future.
    Ugrid = np.zeros([nt,nx]) # Matrix that only receives the results from each time instance at each line.

    ## 1-D linear advection problem
    a = 1.0
    delta_x = x[1] - x[0]   # since it is a linear distribution 
    delta_t = t[1] - t[0]   # From the original inputs, it gives a*delta_t/delta_x == 0.5, which matches the stability conditions
    '''
    #delta_t = 0.5*delta_x/a # 0<= a*delta_t/delta_x <= 1 || Condition for stability  
    print("\nC stab = " + str(a*delta_t/delta_x) + " || cond. for stability: 0 <= a*delta_t/delta_x <= 1\n")
    ## Boundary condition
    print('delta_x :', delta_x)
    print('delta_t :', delta_t)
    print(' ')
    '''
    for ii in range(nx):
        if x[ii] >= xs1 and x[ii] <= xs2:
            Ugrid[0,ii] = 1.0

    ## Analytical solution
    def analytical_sol():
        Outputs = np.zeros([nt,nx])
        for xi in range(nx):
            for ti in range(nt):
                if x[xi]-a*t[ti] >= xs1 and x[xi]-a*t[ti] <= xs2:
                    Outputs[ti][xi] = 1.0
        return Outputs

    Ugrid_ana = analytical_sol()
                
    ############################################################
    ## Execution
    start_time = time.time()

    for ti in range(1,nt): # time sweep, excluding the time at boundary where t == 0
        for xi in range(1,nx): # position sweep, excluding the position where x == -1
            Ugrid[ti,xi] = Ugrid[ti-1][xi] - (a*delta_t/delta_x)*(Ugrid[ti-1][xi] - Ugrid[ti-1][xi-1])

    R_ana    = np.sum((Ugrid-Ugrid_ana)**2)
    MSE_ana  = R_ana/(2*nx*nt) 

    end_time = time.time()
    
    nx = nx + 10
    nt = nt + 10
    if (find_opt == False):
        break

print("\n\n## Solution for the 1-D Linear Advection problem")
print("Convergence time = " + str(end_time-start_time) + " seconds")
print("MSE = " + str(MSE_ana))
print(" ")

nx = nx - 10
nt = nt - 10
############################################################
## Plot the results

level_min=-0.05
level_max=1.05

#Slice arrays
slice_size = 301
Xs      = [[0]*slice_size for _ in range(slice_size)]
Ts      = [[0]*slice_size for _ in range(slice_size)]
Ugrid_s = [[0]*slice_size for _ in range(slice_size)]

ik = 0
for i in np.linspace(0,nx-1,slice_size,dtype=int,endpoint=True):
    jk = 0
    for j in np.linspace(0,nt-1,slice_size,dtype=int,endpoint=True):
        Xs[ik][jk] = X[i][j]
        Ts[ik][jk] = T[i][j]
        Ugrid_s[ik][jk] = Ugrid[i][j]
        jk = jk+1
    ik = ik+1

# Contour plot
fig = plt.figure()
CS = plt.contourf(Xs, Ts, Ugrid_s, cmap=cm.viridis, levels=np.linspace(level_min,level_max,11), extent=[-1, 1, 0, 1])
plt.xlabel(r'$x$',fontsize=18)
plt.ylabel(r'$t$',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
cb = plt.colorbar(CS)
cb.set_label(label=r'$u$',size=18)
cb.ax.tick_params(labelsize=14)
plt.tight_layout()
#plt.show()
fig.savefig("./lin_adv/FD_lin_adv_"+str(nx)+"_x_"+str(nt)+".pdf")

# Figure 2
fig2 = plt.figure(figsize=(8,6))
plt.subplot(2, 1, 1)
plt.plot(x, Ugrid_ana[0][:],'k',label='Analytical',linewidth=2)
plt.plot(x, Ugrid[0],'r',label='Traditional solution',linewidth=2)
plt.legend(loc='upper right', fontsize=18)
plt.title(r'$t=0$', fontsize=18)
plt.ylabel(r'$u$',fontsize=18)
plt.ylabel(r'$u$', fontsize=18)
plt.xlabel(' ',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)

plt.subplot(2, 1, 2)
plt.plot(x, Ugrid_ana[-1][:],'k',linewidth=2)
plt.plot(x, Ugrid[-1],'r',linewidth=2)
plt.title(r'$t=1$', fontsize=18)
plt.ylabel(r'$u$',fontsize=18)
plt.xlabel(r'$x$',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(False)

fig2.tight_layout()
fig2.savefig("./lin_adv/FD_lin_adv_"+str(nx)+"_x_"+str(nt)+"_slice.pdf")
#plt.show()