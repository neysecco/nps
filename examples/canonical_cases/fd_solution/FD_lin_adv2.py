#Traditional solver for the 1-D periodic Linear Advection problem. 
'''
For this problem, we'll be using the first-order Gudonov Method, 
which applies a backward finite difference for the x derivative (for stability reasons) 
and foward difference for the time derivative (since we want to use information from the previous time level). 

I got these equations from the "International Centre for Theoretical Physics" lecture notes.
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

if not os.path.exists('./lin_adv2'):
    os.makedirs('./lin_adv2')
    
#############################################################
## DOMAIN

# Domain sizes
nx = 28001
nt = 28001
nx_ana = 301
nt_ana = 301

## Analytical solution
def analytical_sol(X_ana, T_ana):
    Outputs = np.sin(2*(X_ana-a*T_ana)*np.pi)
    return Outputs

# Bounds
x0 = -1.0
xf = 1.0
t0 = 0.0
tf = 1.0

x = np.linspace(x0,xf,nx)
t = np.linspace(t0,tf,nt)
X,T = np.meshgrid(x,t)

# It might be unnecessary to create a matrix for this problem. But it will be useful in the future.
Ugrid = np.zeros([nt,nx]) # Matrix that only receives the results from each time instance at each line.

## 1-D linear advection problem
a = -0.7
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
    Ugrid[0,ii] = np.sin(2*x[ii]*np.pi)

x_ana = np.linspace(x0,xf,nx_ana)
t_ana = np.linspace(t0,tf,nt_ana)
X_ana,T_ana = np.meshgrid(x_ana,t_ana)
Ugrid_ana = analytical_sol(X_ana, T_ana)

############################################################
## Execution
start_time = time.time()

for ti in range(1,nt): # time sweep, excluding the time at boundary where t == 0
    Ugrid[ti,:-1] = Ugrid[ti-1,:-1] - (a*delta_t/delta_x)*(Ugrid[ti-1][1:] - Ugrid[ti-1][:-1])
    Ugrid[ti,-1] = Ugrid[ti-1,-1] - (a*delta_t/delta_x)*(Ugrid[ti-1][0] - Ugrid[ti-1][-1])

end_time = time.time()

# Interpolate solution to validation points
from scipy.interpolate import RectBivariateSpline as spline
Ugrid_int = spline(t,x,Ugrid).ev(T_ana[:],X_ana[:]).reshape(X_ana.shape)
R_ana    = np.sum((Ugrid_int-Ugrid_ana)**2)
MSE_ana  = R_ana/(2*nx_ana*nt_ana) 

print("\n\n## Solution for the 1-D Periodic Linear Advection problem")
print("Convergence time = " + str(end_time-start_time) + " seconds")
print("MSE = " + str(MSE_ana))
print(" ")
############################################################
## Plot the results

level_min=-1.05
level_max=1.05


#Slice arrays
slice_size = 301
Xs      = [[0]*slice_size for _ in range(slice_size)]
Ts      = [[0]*slice_size for _ in range(slice_size)]
Ugrid_s = [[0]*slice_size for _ in range(slice_size)]

ik = 0
for i in np.linspace(0,nt-1,slice_size,dtype=int,endpoint=True):
    jk = 0
    for j in np.linspace(0,nx-1,slice_size,dtype=int,endpoint=True):
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
fig.savefig("./lin_adv2/FD_lin_adv2_"+str(nx)+"_x_"+str(nt)+".pdf")


# Figure 2
fig2 = plt.figure(figsize=(8,6))
plt.subplot(2, 1, 1)
plt.plot(x_ana, Ugrid_int[0][:],'k',label='Analytical',linewidth=2)
plt.plot(x, Ugrid[0],'r',label='Traditional solution',linewidth=2)
plt.legend(loc='upper right', fontsize=18)
plt.title(r'$t=0$', fontsize=18)
plt.ylabel(r'$u$',fontsize=18)
plt.xlabel(' ',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)

plt.subplot(2, 1, 2)
plt.plot(x_ana, Ugrid_int[-1][:],'k',linewidth=2)
plt.plot(x, Ugrid[-1],'r',linewidth=2)
plt.title(r'$t=1$', fontsize=18)
plt.ylabel(r'$u$',fontsize=18)
plt.xlabel(r'$x$',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid(False)

fig2.tight_layout()
fig2.savefig("./lin_adv2/FD_lin_adv2_"+str(nx)+"_x_"+str(nt)+"_slice.pdf")
#plt.show()