'''
This module contains auxiliary functions
for the linear advection case
'''

# General imports
import numpy as np
from nps import funcs_pde
import matplotlib.pyplot as plt

###########################################

def analytical_sol(Inputs):
    
    # PDE wave speed
    a = 1.0

    # Step position
    xs1 = -0.75
    xs2 = -0.25
    
    Outputs = np.zeros(Inputs.shape[1])

    for ii in range(Inputs.shape[1]):
        x = Inputs[0,ii]
        t = Inputs[1,ii]

        if x-a*t >= xs1 and x-a*t <= xs2:
            Outputs[ii] = 1.0

    return Outputs

###########################################

def compare_plot(NN_set, x0, xf, y0, yf, nx):

    # Select y coordinates for the slices
    y_slices = [y0, yf]

    # Generate pair of slicing points
    Xpair_list = [[[x0, yy], [xf, yy]] for yy in y_slices]

    # The number of points will be equal to the training points
    n_list = [nx]*len(y_slices)

    # Titles
    title_list = [r'$t=%g$'%yy for yy in y_slices]

    # Call the slice function
    figsize = (8,6)

    fig = plot_slice(NN_set,
                     analytical_sol,
                     Xpair_list,
                     n_list,
                     title_list,
                     x_label = r'$x$',
                     p0 = x0,
                     pf = xf,
                     figsize=figsize)

    return fig

###########################################

# Define common function to compute validation metric
def val_function(NN_set, x0, xf, y0, yf):

    # Number of points along one edge of the domain
    nxn = 301

    #Get NNs again
    NN_u = NN_set[0]

    # Generate a refined set of points
    x = np.linspace(x0, xf, nxn)
    y = np.linspace(y0, yf, nxn)
    X,Y = np.meshgrid(x,y)
    Inputs = np.vstack([X.ravel(),Y.ravel()])

    # Evaluate using the ANN
    Outputs_T = NN_u.feedforward(Inputs)[0]

    # Analytical solution
    Outputs = analytical_sol(Inputs)

    # Compute MSE over the validation set
    valMSE = np.sum((Outputs_T - Outputs)**2)/2/len(Outputs)

    return valMSE

###########################################

def MSEhist_function(NN_set, case_name, Theta_hist_ALM,
                     Residual_PDE, Residual_BC,
                     x0, xf, y0, yf):

    # Get the number of points
    num_PDE     = len(Residual_PDE(NN_set)[0])
    num_BC      = len(Residual_BC(NN_set)[0])
    num_weights = len(Theta_hist_ALM[0])

    # Transpose matrix to facilitate for loop
    # Now each weight iteration is a line
    Theta_hist_ALM = Theta_hist_ALM.T

    # Initialize arrays to hold history results
    valMSE_ALM         = np.zeros(len(Theta_hist_ALM))
    Residual_PDE_ALM   = np.zeros(len(Theta_hist_ALM))
    Residual_BC_ALM    = np.zeros(len(Theta_hist_ALM))
    weights_ALM        = np.zeros(len(Theta_hist_ALM))
    iterations_ALM     = np.arange(len(Theta_hist_ALM))

    # Loop over every time instance
    for j in range(len(Theta_hist_ALM)):

        # Assing weights to the ANN
        funcs_pde.reassign_theta(NN_set,Theta_hist_ALM[j])

        valMSE_ALM[j]        = val_function(NN_set, x0, xf, y0, yf)
        print('MSEval at the '+ str(j) + '-th ALM iteration = ' + str(valMSE_ALM[j]))

        Residual_PDE_ALM[j]  = np.sum(Residual_PDE(NN_set)[0]**2)/2/num_PDE
        Residual_BC_ALM[j]   = np.sum(Residual_BC(NN_set)[0]**2)/2/num_BC
        weights_ALM[j]       = np.sum(Theta_hist_ALM[j]**2)/2/num_weights

    ## MSE evolution over ALM iterations
    fig2 = plt.figure()
    plt.semilogy(iterations_ALM,valMSE_ALM, '-o', linewidth=2, label=r'$MSE_{val}$')
    plt.semilogy(iterations_ALM,Residual_PDE_ALM, '-o', linewidth=2, label=r'PDE residuals')
    plt.semilogy(iterations_ALM,Residual_BC_ALM, '-o', linewidth=2, label=r'BC residuals')
    plt.semilogy(iterations_ALM,weights_ALM, '-o', linewidth=2, label=r'ANN weights')
    plt.legend(loc='best',fontsize=18)

    # Add labels
    plt.xlabel(r'$\mathrm{ALM \; iterations}$',fontsize=18)
    plt.ylabel(r'$\mathrm{Mean \; Squared \; Value}$',fontsize=18)
    plt.xticks(iterations_ALM,fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    # Save figure
    fig2.savefig('./' + case_name + '/' + case_name + '_MSE_ALM.pdf',dpi=300)
    plt.close(fig2)

###########################################

def plot_surface(NN_set,
                 x0, xf, y0, yf, nx, ny,
                 Inputs_bc, Targets_bc, Inputs_do,
                 level_min=-0.05, level_max=1.05,axis_label=['x','y']):

    '''
    This function plots a surface plot for the given ANN and BCs

    level_min and level_max are the bounds for the contour plot.
    '''

    # Get NNs again
    NN_u = NN_set[0]

    # Generate input points and evaluate ANN

    x = np.linspace(x0,xf,nx*3)
    y = np.linspace(y0,yf,ny*3)
    X,Y = np.meshgrid(x,y)

    Inputs = np.vstack([X.ravel(),Y.ravel()])
    Outputs = NN_u.feedforward(Inputs)[0]
    Z = Outputs.reshape(X.shape)

    #More imports
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm, rcParams, pyplot
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    '''
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.family'] = 'STIXGeneral'
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf1 = ax.plot_surface(X, Y, Z_T, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.scatter(Inputs_bc[0,:],Inputs_bc[1,:],Targets_bc)
    fig.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    '''

    # Contour plot
    fig2 = plt.figure()
    CS = plt.contourf(X, Y, Z, cmap=cm.viridis, levels=np.linspace(level_min,level_max,11))
    plt.plot(Inputs_do[0,:],Inputs_do[1,:],'o',markersize=3,c='r',alpha=0.5)
    plt.plot(Inputs_bc[0,:],Inputs_bc[1,:],'o',markersize=3,c='r',alpha=0.5)
    plt.xlabel(r'$' + axis_label[0] + '$',fontsize=20)
    plt.ylabel(r'$' + axis_label[1] + '$',fontsize=20)
    cb = plt.colorbar(CS)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$u$',size=20)
    fig2.gca().spines['right'].set_visible(True)
    fig2.gca().spines['top'].set_visible(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #fig2.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    plt.tight_layout()

    # RETURNS
    return fig2 # The user should return the figure handle

###########################################

def plot_slice(NN_set,
               analytical_sol,
               Xpair_list,
               n_list,
               title_list,
               x_label = r'$\alpha$',
               figsize=(8,6),
               p0 = 0,
               pf = 1,
               legend_box = 1):
    '''
    This functions plots a figure containing
    several domain slices

    INPUTS:
    Xpair_list: list -> List of initial and final points of the slices
                        Xpair_list = [[[xs1, ys1], [xf1, yf1]],
                                      [[xs2, ys2], [xf2, yf2]],
                                      [[xs3, ys3], [xf3, yf3]],
                                      .........................]
    legend_box: which subplot should have the legend box
    '''

    # Get reference to the ANN
    NN_u = NN_set[0]

    # Get the total number of slices
    num_slices = len(n_list)

    # Inicialize slice counter
    slice_id = 1

    # Initialize figure
    fig = plt.figure(figsize=figsize)

    # Loop over the coordinate pairs to generate each subplot
    for Xpair, n, title in zip(Xpair_list, n_list, title_list):

        # Get initial coordinate
        Xs = Xpair[0]

        # Get final coordinate
        Xf = Xpair[1]

        # Build an array of points along the line
        Inputs = np.zeros((2, n))
        alpha = np.linspace(0,1,n)

        Inputs[0,:] = Xs[0] + alpha*(Xf[0]-Xs[0])
        Inputs[1,:] = Xs[1] + alpha*(Xf[1]-Xs[1])

        alpha = np.linspace(p0,pf,n)
        # Build an array of refined points along the same line
        ref_factor = 5
        Inputs_ref = np.zeros((2, n*ref_factor))
        alpha_ref = np.linspace(0,1,n*ref_factor)
        Inputs_ref[0,:] = Xs[0] + alpha_ref*(Xf[0]-Xs[0])
        Inputs_ref[1,:] = Xs[1] + alpha_ref*(Xf[1]-Xs[1])

        # Get the ANN solution at the original points
        Outputs = NN_u.feedforward(Inputs)[0]

        # Get the ANN solution at the refined points
        Outputs_ref = NN_u.feedforward(Inputs_ref)[0]

        # Get the analytical solution at the refined points
        Outputs_ref_an = analytical_sol(Inputs_ref)

        alpha = np.linspace(p0,pf,n)
        alpha_ref = np.linspace(p0,pf,n*ref_factor)
        # PLOT

        # Select subplot
        plt.subplot(num_slices, 1, slice_id)

        # Plot analytical solution
        plt.plot(alpha_ref, Outputs_ref_an, 'k', label=r'$\mathrm{Analytical}$', linewidth=2)

        # Plot ANN solution
        plt.scatter(alpha, Outputs, c='r', label=r'$\mathrm{ANN \; training \; points}$')

        # Plot ANN solution at refined points
        plt.plot(alpha_ref, Outputs_ref, 'r', linewidth=2, label=r'$\mathrm{ANN}$')

        # Add title to the plot
        plt.title(r'' + title, fontsize=22)
        plt.ylabel(r'$u$',fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # Add legend to the first plot only
        if slice_id == legend_box:
            plt.legend(loc='best',fontsize=18)

        # Increment slice counter
        slice_id = slice_id + 1
        #fig.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places

    # Add label to the horizontal axis
    plt.xlabel(r''+ x_label,fontsize=22)

    # Adjust the plot
    plt.tight_layout()

    return fig

###########################################
