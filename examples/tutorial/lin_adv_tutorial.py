'''
This script uses NPS to solve the linear advection case for u(x,t):
du/dt  + a*du/dx = 0
The problem is defined is defined in a rectangular domain:
x = [-1, 1]
t = [ 0, 1]
Boundary conditions are:
u(x,0) = square wave along
u(0,t) = 0
'''

from nps import NN, funcs_pde
import numpy as np
import os
import aux_mod as am

#=====================================

## INPUTS

case_name = 'lin_adv'

optimizer = 'ALM' # options: 'ALM' or 'slsqp'

# Domain bounds
x0 = -1.0
xf = 1.0
y0 = 0.0
yf = 1.0

# Number of boundary points
num_x = 41
num_y = 41

# Number of interior points randomly distributed within the domain
num_int = 100

# PDE wave speed
a = 1.0

# Step position
xs1 = -0.75
xs2 = -0.25

# Number of neurons in each hidden layer (you can add more hidden layers if necessary)
num_neurons = [3,3]

#=====================================

## SETUP

# Define a random number generator for repeatibility
rng_seed = 123
rng = np.random.default_rng(seed=rng_seed)

# Define files and directories
hist_file = case_name+'/results.pickle'
image_dir = './'+case_name

# Check if the images directory exists
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Wipe out everything inside the output directory
os.system('rm ' + image_dir+'/*.png')
os.system('rm ' + image_dir+'/*.pdf')

#=====================================

## DOMAIN

# Generate boundary points
x_side = np.linspace(x0, xf, num_x)
y_side = np.linspace(y0, yf, num_y)

# Build (x,t) pairs of boundary points. Each sampling point is a column.
Inputs_lower = np.vstack([x_side, y0*np.ones(num_x)]) # points along y=0
Inputs_left = np.vstack([x0*np.ones(num_y-1), y_side[1:]]) # points along x=0
Inputs_upper = np.vstack([x_side, yf*np.ones(num_x)]) # points along y=1
Inputs_right = np.vstack([xf*np.ones(num_y-1), y_side[1:]]) # points along x=1

# Interior points
x_int = (xf-x0)*rng.random(num_int) + x0
y_int = (yf-y0)*rng.random(num_int) + y0

Inputs_int = np.vstack([x_int,y_int])

# The domain points where we enforce the PDE
# must include interior and boundary points
Inputs_do = np.hstack([Inputs_int, Inputs_lower, Inputs_right, Inputs_upper, Inputs_left])

#=====================================

## NEURAL NETWORK

# Create one neural network for the field variable
# to represent u = u(x,t)
NN_u = NN(num_inputs=2, num_outputs=1,
          num_neurons_list=num_neurons,
          layer_type=['sigmoid']*len(num_neurons),
          Theta_flat=rng_seed,
          lower_bounds=[x0,y0],upper_bounds=[xf,yf])

# Create list of ANNs (you can define multiple ANNs for the same problem)
NN_set = [NN_u]

## BOUNDARY CONDITIONS

# Step for the bottom edge (y=0) and u=0 over the left side (x=0)
Targets_lower = np.zeros(Inputs_lower.shape[1])
for index in range(Inputs_lower.shape[1]):
    if Inputs_lower[0,index] >= xs1 and Inputs_lower[0,index] <= xs2:
        Targets_lower[index] = 1

Targets_left = 0.0*np.ones(Inputs_left.shape[1])

Inputs_bc = np.hstack([Inputs_lower, Inputs_left])
Targets_bc = np.hstack([Targets_lower, Targets_left])

## RESIDUAL FUNCTIONS

def Residual_BC(NN_set):

    # This function computes the BC residuals at the boundary points for multiple networks.
    # It also gives the residuals gradients with respect to the weights.
    # For this problem I'll use
    # NN_set[0]: NN_u

    # Use all NNs (even though we have only one for this example)
    Outputs_set, dOutputs_dTheta_set = funcs_pde.use_NN_set(Inputs_bc,NN_set)[0:2]
    Outputs_bc = Outputs_set[0]
    dOutputs_dTheta = dOutputs_dTheta_set[0]

    #Compute the total residual
    ResBC = Outputs_bc-Targets_bc

    #Compute the residual derivatives (differentiate ResBC w.r.t. Theta)
    dResBC_dTheta = dOutputs_dTheta

    # Returns
    return ResBC, dResBC_dTheta

###########################################

def Residual_PDE(NN_set):

    # First variable is x, second is y (x1=x, x2=y)
    # For this problem I'll use
    # NN_set[0]: NN_u

    # Use all NNs
    Sens_set, dSens_dTheta_set = funcs_pde.use_NN_set(Inputs_do,NN_set)[2:4]

    # Compute Residual
    # res = dudy + a*dudx = 0
    ResPDE = Sens_set[0][1,:] + a*Sens_set[0][0,:]

    # Residuals gradient
    dResPDE_dTheta = dSens_dTheta_set[0][:,1,:] + a*dSens_dTheta_set[0][:,0,:]

    # Returns
    return ResPDE, dResPDE_dTheta

###########################################

# Optimization parameters
metric = 'MSE' # options: 'MSE' or 'KS'
reg_factor = 0.0 # regularization factor to avoid numerical issues for high weights
rhoKS = 0.5 # Only used for 'KS' metric
opt_options = {}

if optimizer == 'ALM':

    opt_options = {'grad_opt':'BFGS', # options: 'fminscg' or 'BFGS'
                   'display':True,
                   'major_iterations':5,
                   'gamma':1.,
                   'delta_x_tol':1e-5,
                   'minor_iterations':3000,
                   'grad_tol':1e-5,
                   'save_minor_hist':False,
                   'forced_iterations':0}

elif optimizer == 'slsqp':

    opt_options = {'ftol':1e-9,
                   'disp':True,
                   'iprint':2,
                   'maxiter':1000}

#==========================================

# Define common plot_function for runtime verification
# This function will be called after a given number of iterations
# along the training, so the user can verify the optimization progress

def plot_function(NN_set):
    
    # Contour plot levels
    level_min = -0.05
    level_max = 1.05
    
    # Use helper function
    fig1 = am.plot_surface(NN_set,
                           x0, xf, y0, yf, num_x, num_y,
                           Inputs_bc, Targets_bc, Inputs_do,
                           level_min, level_max,
                           axis_label=['x','t'])
    
    # Generate particular plot of the case
    fig2 = am.compare_plot(NN_set,
                           x0, xf, y0, yf, num_x)
            
    # RETURNS
    return [fig1, fig2] # The user should return a list of figure handles

#==========================================

# Solve PDE
[obj_hist,
 Theta_flat_hist,
 training_time,
 Theta_hist_ALM] = funcs_pde.train_pde_nn(NN_set,
                                          Residual_BC, 
                                          Residual_PDE,
                                          optimizer = optimizer,
                                          metric = metric,
                                          opt_options = opt_options,
                                          reg_factor = reg_factor,
                                          rhoKS = rhoKS,
                                          check_gradients = True,
                                          hist_file = hist_file,
                                          load_hist = False,
                                          plot_function = plot_function,
                                          image_dir = image_dir)

# Plot optimization history
am.MSEhist_function(NN_set, case_name, Theta_hist_ALM,
                    Residual_PDE, Residual_BC, x0, xf, y0, yf)                                                                   
