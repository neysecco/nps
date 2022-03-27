from nps import NN, funcs_pde, postproc_pde
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams, pyplot
from matplotlib.ticker import LinearLocator, FormatStrFormatter,StrMethodFormatter

#rcParams['mathtext.fontset'] = 'stix'
#rcParams['font.family'] = 'STIXGeneral'

def load_case(case_name, dist_type, optimizer, boundary_points, domain_points,
              num_neurons=None, clean_image_dir=False, Theta_seed=123, axis_label=['x','y']):
    '''
    This is the main interface to load PDE problems.

    INPUTS:
    case_name: string -> Name of the PDE problem. Current options are:
                         'heat_eq'.

    dist_type: string -> Distribution of domain points. Current options are:
                         'structured' or 'unstructured'.

    num_neurons: list of integers -> Number of neurons at the hidden layers.
                                     Example: [10,5]
                                     If None, then default values will be used
                                     for each test case.

    clean_image_dir: boolean -> Flag to remove all previous files in the
                                image_dir folder of the test_case.
    '''

    # Define a random number generator for repeatibility
    rng = np.random.default_rng(seed=123)

    opt_options = {}

    if case_name in ['lin_adv','lin_adv2','lin_adv_aneg']:

        ## DOMAIN

        # Bounds
        x0 = -1.0
        xf = 1.0
        y0 = 0.0
        yf = 1.0

        # Generate refined boundaries
        nx = boundary_points
        ny = boundary_points

        # Step position
        xs1 = -0.75
        xs2 = -0.25
        if case_name == 'lin_adv_aneg':
            xs1 = 0.25
            xs2 = 0.75
            
        # PDE wave speed and number of neurons

        if case_name == 'lin_adv':
            
            a = 1.0
            if num_neurons is None:
                num_neurons = [3, 2]

        elif case_name == 'lin_adv2':

            a = -0.7
            if num_neurons is None:
                num_neurons = [3, 2]
               
        elif case_name == 'lin_adv_aneg':
        
            a = -1.0
            if num_neurons is None:
                num_neurons = [3, 2]
                
        [Inputs_lower,
         Inputs_right,
         Inputs_top,
         Inputs_left] = create_2Ddomain(x0, xf, nx,
                                        y0, yf, ny,
                                        dist=dist_type,
                                        rng=rng)[1:]

        # Generate coarse interior points
        nxint = domain_points
        nyint = domain_points

        Inputs_do = create_2Ddomain(x0, xf, nxint,
                                    y0, yf, nyint,
                                    dist=dist_type,
                                    no_borders=True,
                                    rng=rng)[0]

        # Add the refined boundary points to the coarse domain
        Inputs_do = np.hstack([Inputs_do, Inputs_lower, Inputs_right, Inputs_top, Inputs_left])

        ## NEURAL NETWORK

        # Create one neural network for the field variable
        NN_u = NN(num_inputs=2, num_outputs=1, num_neurons_list=num_neurons,
                  layer_type=['sigmoid']*len(num_neurons), Theta_flat=Theta_seed,
                  lower_bounds=[x0,y0],upper_bounds=[xf,yf])

        NN_set = [NN_u]

        ## BOUNDARY CONDITIONS
        
        # Step for the bottom edge (y=0) and u=0 over the left side (x=0)

        if (case_name == 'lin_adv' or case_name == 'lin_adv_aneg'):

            Targets_lower = np.zeros(Inputs_lower.shape[1])
            for index in range(Inputs_lower.shape[1]):
                if Inputs_lower[0,index] >= xs1 and Inputs_lower[0,index] <= xs2:
                    Targets_lower[index] = 1

            # Join all boundary points
            if case_name == 'lin_adv':
                Targets_left = 0.0*np.ones(Inputs_left.shape[1])
                Inputs_bc = np.hstack([Inputs_lower, Inputs_left])
                Targets_bc = np.hstack([Targets_lower, Targets_left])
            elif case_name == 'lin_adv_aneg':
                Targets_right = 0.0*np.ones(Inputs_right.shape[1])
                Inputs_bc = np.hstack([Inputs_lower, Inputs_right])
                Targets_bc = np.hstack([Targets_lower, Targets_right])
            
        elif case_name == 'lin_adv2':

            # We will add a sine wave at the lower edge and
            # assign periodic bc at the sides
            Targets_lower = np.sin(2*Inputs_lower[0,:]*np.pi)
    
            # Join all boundary points
            Inputs_bc = Inputs_lower
            Targets_bc = Targets_lower

        ## RESIDUAL FUNCTIONS

        def Residual_BC(NN_set):

            # This function computes the residuals at the given points for multiple networks.
            # It also gives the residuals gradients with respect to the weights.

            #For this problem I'll use
            #NN_set[0]: NN_u

            #Use all NNs (even though we have only one for this example)
            Outputs_set, dOutputs_dTheta_set = funcs_pde.use_NN_set(Inputs_bc,NN_set)[0:2]
            Outputs_bc = Outputs_set[0]
            dOutputs_dTheta = dOutputs_dTheta_set[0]

            #Compute the total residual
            ResBC = Outputs_bc-Targets_bc

            #Compute the MSE derivative
            dResBC_dTheta = dOutputs_dTheta

            # Add periodic BCs for the second case
            if case_name == 'lin_adv2':

                Outputs_set, dOutputs_dTheta_set = funcs_pde.use_NN_set(Inputs_left,NN_set)[0:2]
                Outputs_left = Outputs_set[0]
                dOutputs_dTheta_left = dOutputs_dTheta_set[0]

                Outputs_set, dOutputs_dTheta_set = funcs_pde.use_NN_set(Inputs_right,NN_set)[0:2]
                Outputs_right = Outputs_set[0]
                dOutputs_dTheta_right = dOutputs_dTheta_set[0]

                #Compute the total residual
                ResBC2 = Outputs_left-Outputs_right

                #Compute the MSE derivative
                dResBC2_dTheta = dOutputs_dTheta_left - dOutputs_dTheta_right

                # Join results
                ResBC = np.hstack([ResBC, ResBC2])
                dResBC_dTheta = np.hstack([dResBC_dTheta, dResBC2_dTheta])

            #RETURNS
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

        def analytical_sol(Inputs):

            Outputs = np.zeros(Inputs.shape[1])

            for ii in range(Inputs.shape[1]):

                x = Inputs[0,ii]
                t = Inputs[1,ii]

                if (case_name == 'lin_adv' or case_name == 'lin_adv_aneg'):

                    if x-a*t >= xs1 and x-a*t <= xs2:
                        Outputs[ii] = 1.0

                elif case_name == 'lin_adv2':

                    Outputs[ii] = np.sin(2*(x-a*t)*np.pi)

            return Outputs

        ###########################################

        def compare_plot(NN_set):

            # Select y coordinates for the slices
            y_slices = [y0, yf]

            # Generate pair of slicing points
            Xpair_list = [[[x0, yy], [xf, yy]] for yy in y_slices]

            # The number of points will be equal to the training points
            n_list = [nx]*len(y_slices)

            # Titles
            title_list = [r'$t=%g$'%yy for yy in y_slices]

            # Call the slice function
            fig = plot_slice(NN_set,
                             analytical_sol,
                             Xpair_list,
                             n_list,
                             title_list,
                             x_label = r'$x$',
                             p0 = x0,
                             pf = xf)

            return [fig]

        # Levels for the contour plot
        if (case_name == 'lin_adv' or case_name == 'lin_adv_aneg'):
            level_min = -0.05
            level_max = 1.05
        elif case_name == 'lin_adv2':
            level_min = -1.05
            level_max = 1.05
            
        ###########################################


        # Optimization parameters
        reg_factor = 0.0
        rhoKS = 0.5
        opt_options = {}

        if optimizer == 'ALM':

            opt_options = {'grad_opt':'fminscg',
                           'display':True,
                           'major_iterations':10,
                           'gamma':1.,
                           'delta_x_tol':1e-10,
                           'minor_iterations':3000,
                           'grad_tol':1e-10,
                           'save_minor_hist':False,
                           'forced_iterations':0}

        elif optimizer == 'slsqp':
        
            opt_options = {'ftol':1e-9,
                           'disp':True,
                           'iprint':2,
                           'maxiter':1000}

    ###########################################
    ###########################################

    elif case_name == 'heat_eq':

        ## DOMAIN
        x0 = 0.0
        xf = 1.0
        y0 = 0.0
        yf = 1.0

        # OPTION 1
        # Generate refined boundaries
        nx = boundary_points
        ny = boundary_points
        
        [Inputs_lower,
         Inputs_right,
         Inputs_top,
         Inputs_left] = create_2Ddomain(x0, xf, nx,
                                        y0, yf, ny,
                                        dist=dist_type,
                                        rng=rng)[1:]
                                        
        # Generate coarse interior points
        nxint = domain_points
        nyint = domain_points

        Inputs_do = create_2Ddomain(x0, xf, nxint,
                                    y0, yf, nyint,
                                    dist=dist_type,
                                    no_borders=True,
                                    rng=rng)[0]

        # Add the refined boundary points to the coarse domain
        Inputs_do = np.hstack([Inputs_do, Inputs_lower, Inputs_right, Inputs_top, Inputs_left])
        
        
        '''
        # OPTION 2
        # Generate refined boundaries
        nx = 10
        ny = 10
        
        [Inputs_do,
         Inputs_lower,
         Inputs_right,
         Inputs_top,
         Inputs_left] = create_2Ddomain(x0, xf, nx,
                                        y0, yf, ny,
                                        dist=dist_type,
                                        rng=rng)
        '''
        
        ## NEURAL NETWORK

        # Create one neural network for the field variable
        if num_neurons is None:
            num_neurons = [7, 7]
        NN_u = NN(num_inputs=2, num_outputs=1, num_neurons_list=num_neurons,
                  layer_type=['sigmoid']*len(num_neurons), Theta_flat=Theta_seed,
                  lower_bounds=[x0,y0],upper_bounds=[xf,yf])

        NN_set = [NN_u]

        ## BOUNDARY CONDITIONS
        
        # u=1 for y=0 edge and u=0 over all other edges

        Targets_lower = 1.0*np.ones(Inputs_lower.shape[1])
        Targets_lower[0] = 0.
        Targets_lower[-1] = 0.
        
        Targets_left = 0.0*np.ones(Inputs_left.shape[1])
        
        Targets_top = 0.0*np.ones(Inputs_top.shape[1])
        
        Targets_right = 0.0*np.ones(Inputs_right.shape[1])
        
        # Join all boundary points
        Inputs_bc = np.hstack([Inputs_lower, Inputs_left, Inputs_top, Inputs_right])
        Targets_bc = np.hstack([Targets_lower, Targets_left, Targets_top, Targets_right])

        ## RESIDUAL FUNCTIONS
         
        def Residual_BC(NN_set):

            # This function computes the residuals at the given points for multiple networks.
            # It also gives the residuals gradients with respect to the weights.

            #For this problem I'll use
            #NN_set[0]: NN_T

            #Use all NNs (even though we have only one for this example)
            Outputs_set, dOutputs_dTheta_set = funcs_pde.use_NN_set(Inputs_bc,NN_set)[0:2]
            Outputs_bc = Outputs_set[0]
            dOutputs_dTheta = dOutputs_dTheta_set[0]

            #Compute the total residual
            ResBC = Outputs_bc-Targets_bc

            #Compute the MSE derivative
            dResBC_dTheta = dOutputs_dTheta

            #RETURNS
            return ResBC, dResBC_dTheta

        ###########################################

        def Residual_PDE(NN_set):
        
            #First variable is x, second is y (x1=x, x2=y)
            #For this problem I'll use
            #NN_set[0]: NN_u

            #Use all NNs
            Hess_set, dHess_dTheta_set = funcs_pde.use_NN_set(Inputs_do,NN_set)[4:]

            #Compute Residual
            #d2Tdx2 + d2Tdy2 = 0
            ResPDE = Hess_set[0][0,0,:] + Hess_set[0][1,1,:]

            #Residuals gradient
            dResPDE_dTheta = dHess_dTheta_set[0][:,0,0,:] + dHess_dTheta_set[0][:,1,1,:]

            #Returns
            return ResPDE, dResPDE_dTheta    
        
        ###########################################

        def analytical_sol(Inputs):

            # Number of terms in the Fourier Series
            n_fourier = 150

            # Initializing grid to accumulate the terms of the Fourier series
            Outputs = np.zeros(Inputs.shape[1])

            for  n in range(1,n_fourier+1):
                delta = 2.*((-1.)**n-1.)/n/np.pi/np.sinh(n*np.pi)*np.sin(n*np.pi*Inputs[0,:])*np.sinh(n*np.pi*(Inputs[1,:]-1.))
                Outputs = Outputs + delta #Increment of the grid

            return Outputs

        ###########################################

        def compare_plot(NN_set):

            # Select x coordinates for the slices
            x_slices = [0.2, 0.50, 0.8]

            # Generate pair of slicing points
            Xpair_list = [[[xx, y0], [xx, yf]] for xx in x_slices]

            # The number of points will be equal to the training points
            n_list = [ny]*len(x_slices)

            # Titles
            title_list = [r'$x=%g$'%xx for xx in x_slices]

            # Call the slice function
            fig = plot_slice(NN_set,
                             analytical_sol,
                             Xpair_list,
                             n_list,
                             title_list,
                             x_label = r'$y$',
                             figsize=(8,8),
                             p0 = y0,
                             pf = yf)

            return [fig]

        # Levels for the contour plot
        level_min = -0.05
        level_max = 1.05

        ###########################################

        # Optimization parameters
        reg_factor = 0.00
        rhoKS = 0.5

        if optimizer == 'ALM':

            opt_options = {'grad_opt':'bfgs',
                           'display':True,
                           'major_iterations':10,
                           'gamma':1.,
                           'delta_x_tol':1e-5,
                           'minor_iterations':3000,
                           'grad_tol':1e-5,
                           'save_minor_hist':False,
                           'forced_iterations':0}

        elif optimizer == 'slsqp':
        
            opt_options = {'ftol':1e-7,
                           'disp':True,
                           'iprint':2,
                           'maxiter':10000}

    ###########################################
    ###########################################

    elif case_name == 'heat_cond':

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

        ## DOMAIN

        # Bounds
        x0 = 0.0
        xf = L
        y0 = 0.0
        yf = tf

        # Generate refined boundaries
        nx = boundary_points
        ny = boundary_points

        [Inputs_lower,
         Inputs_right,
         Inputs_top,
         Inputs_left] = create_2Ddomain(x0, xf, nx,
                                        y0, yf, ny,
                                        dist=dist_type,
                                        rng=rng)[1:]

        # Generate coarse interior points
        nxint = domain_points
        nyint = domain_points

        Inputs_do = create_2Ddomain(x0, xf, nxint,
                                    y0, yf, nyint,
                                    dist=dist_type,
                                    no_borders=True,
                                    rng=rng)[0]

        # Add the refined boundary points to the coarse domain
        Inputs_do = np.hstack([Inputs_do, Inputs_lower, Inputs_right, Inputs_top, Inputs_left])

        ## NEURAL NETWORK

        # Create one neural network for the field variable
        if num_neurons is None:
            num_neurons = [5, 5]
        NN_u = NN(num_inputs=2, num_outputs=1, num_neurons_list=num_neurons,
                  layer_type=['sigmoid']*len(num_neurons), Theta_flat=Theta_seed,
                  lower_bounds=[x0,y0],upper_bounds=[xf,yf])

        NN_set = [NN_u]

        ## BOUNDARY CONDITIONS
        
        # u=1 for y=0 edge and u=0 over all other edges

        Targets_lower = Tr*np.ones(Inputs_lower.shape[1])
        Targets_lower[0] = Tl
        
        Targets_left = Tl*np.ones(Inputs_left.shape[1])

        Targets_right = Tr*np.ones(Inputs_right.shape[1])

        # Join all boundary points
        Inputs_bc = np.hstack([Inputs_lower, Inputs_left, Inputs_right])
        Targets_bc = np.hstack([Targets_lower, Targets_left, Targets_right])

        ## RESIDUAL FUNCTIONS

        def Residual_BC(NN_set):

            # This function computes the residuals at the given points for multiple networks.
            # It also gives the residuals gradients with respect to the weights.

            #For this problem I'll use
            #NN_set[0]: NN_u

            #Use all NNs (even though we have only one for this example)
            Outputs_set, dOutputs_dTheta_set = funcs_pde.use_NN_set(Inputs_bc,NN_set)[0:2]
            Outputs_bc = Outputs_set[0]
            dOutputs_dTheta = dOutputs_dTheta_set[0]

            #Compute the total residual
            ResBC = Outputs_bc-Targets_bc

            #Compute the MSE derivative
            dResBC_dTheta = dOutputs_dTheta

            #RETURNS
            return ResBC, dResBC_dTheta

        ###########################################

        def Residual_PDE(NN_set):
        
            # First variable is x, second is y (x1=x, x2=y)
            # For this problem I'll use
            # NN_set[0]: NN_u

            # Use all NNs
            Sens_set, dSens_dTheta_set, Hess_set, dHess_dTheta_set = funcs_pde.use_NN_set(Inputs_do,NN_set)[2:]

            # Compute Residual
            #res = dudy - k*d2udx2 = 0
            ResPDE = Sens_set[0][1,:] - k*Hess_set[0][0,0,:]

            # Residuals gradient
            dResPDE_dTheta = dSens_dTheta_set[0][:,1,:] - k*dHess_dTheta_set[0][:,0,0,:]

            # Returns
            return ResPDE, dResPDE_dTheta    

        ###########################################

        # ANALYTICAL SOLUTION

        def analytical_sol(Inputs):

            # Initialize Output array
            Outputs = np.zeros(Inputs.shape[1])

            # Temperature difference
            deltaT = Tl - Tr

            # Select number of Fourier terms
            N = np.arange(1,501)

            # Fourier coefficients
            bn = L*deltaT/np.pi**2/N**2*(np.sin(np.pi*N) - np.pi*N)

            for ii in range(len(Outputs)):

                x = Inputs[0,ii]
                t = Inputs[1,ii]

                Outputs[ii] = deltaT*(1-x/L) + 2/L*np.sum(bn*np.sin(N*np.pi*x/L)*np.exp(-k*N**2*np.pi**2*t/L**2))

            return Outputs

        ###########################################

        def compare_plot(NN_set):

            # Select y coordinates for the slices
            y_slices = [y0, 0.5*yf, yf]

            # Generate pair of slicing points
            Xpair_list = [[[x0, yy], [xf, yy]] for yy in y_slices]

            # The number of points will be equal to the training points
            n_list = [nx]*len(y_slices)

            # Titles
            title_list = [r'$t=%g$'%yy for yy in y_slices]

            # Call the slice function
            fig = plot_slice(NN_set,
                             analytical_sol,
                             Xpair_list,
                             n_list,
                             title_list,
                             x_label = r'$x$',
                             figsize=(8,8),
                             p0 = x0,
                             pf = xf)

            return [fig]

        # Levels for the contour plot
        level_min = -0.05
        level_max = 1.05

        ###########################################


        # Optimization parameters
        reg_factor = 0.0
        rhoKS = 0.5
        opt_options = {}

        if optimizer == 'ALM':

            opt_options = {'grad_opt':'bfgs',
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
                           'maxiter':10000}

    ###########################################
    ###########################################

    elif case_name == 'burgers':

        ## DOMAIN

        # Bounds
        x0 = 0.0
        xf = 1.0
        y0 = 0.0
        yf = 1.0

        # Generate refined boundaries
        nx = boundary_points
        ny = boundary_points

        [Inputs_lower,
         Inputs_right,
         Inputs_top,
         Inputs_left] = create_2Ddomain(x0, xf, nx,
                                        y0, yf, ny,
                                        dist=dist_type,
                                        rng=rng)[1:]

        # Generate coarse interior points
        nxint = domain_points
        nyint = domain_points

        Inputs_do = create_2Ddomain(x0, xf, nxint,
                                    y0, yf, nyint,
                                    dist=dist_type,
                                    no_borders=True,
                                    rng=rng)[0]

        # Add the refined boundary points to the coarse domain
        Inputs_do = np.hstack([Inputs_do, Inputs_lower, Inputs_right, Inputs_top, Inputs_left])

        ## NEURAL NETWORK

        # Create one neural network for the field variable
        if num_neurons is None:
            num_neurons = [5, 5]
        NN_u = NN(num_inputs=2, num_outputs=1, num_neurons_list=num_neurons,
                  layer_type=['sigmoid']*len(num_neurons), Theta_flat=Theta_seed,
                  lower_bounds=[x0,y0],upper_bounds=[xf,yf])

        NN_set = [NN_u]

        ## BOUNDARY CONDITIONS
        
        # Bump for the bottom edge (y=0) and u=0.1 over the left side (x=0)

        Targets_lower = 0.1*np.ones(Inputs_lower.shape[1])
        for index in range(Inputs_lower.shape[1]):
            if Inputs_lower[0,index] > 0.3 and Inputs_lower[0,index] < 0.4:
                Targets_lower[index] = 0.1+0.1*(Inputs_lower[0,index]-0.3)/(0.4-0.3)
            elif Inputs_lower[0,index] >= 0.4 and Inputs_lower[0,index] <= 0.6:
                Targets_lower[index] = 0.2
            elif Inputs_lower[0,index] > 0.6 and Inputs_lower[0,index] < 0.7:
                Targets_lower[index] = 0.2-0.1*(Inputs_lower[0,index]-0.6)/(0.7-0.6)
        
        Targets_left = 0.1*np.ones(Inputs_left.shape[1])
        
        # Join all boundary points
        Inputs_bc = np.hstack([Inputs_lower, Inputs_left])
        Targets_bc = np.hstack([Targets_lower, Targets_left])

        ## RESIDUAL FUNCTIONS

        def Residual_BC(NN_set):

            # This function computes the residuals at the given points for multiple networks.
            # It also gives the residuals gradients with respect to the weights.

            #For this problem I'll use
            #NN_set[0]: NN_T

            #Use all NNs (even though we have only one for this example)
            Outputs_set, dOutputs_dTheta_set = funcs_pde.use_NN_set(Inputs_bc,NN_set)[0:2]
            Outputs_bc = Outputs_set[0]
            dOutputs_dTheta = dOutputs_dTheta_set[0]

            #Compute the total residual
            ResBC = Outputs_bc-Targets_bc

            #Compute the MSE derivative
            dResBC_dTheta = dOutputs_dTheta

            #RETURNS
            return ResBC, dResBC_dTheta

        ###########################################

        def Residual_PDE(NN_set):
        
            #First variable is x, second is t (x1=x, x2=t)
            #For this problem I'll use
            #NN_set[0]: NN_y

            #Use all NNs
            Outputs_set, dOutputs_dTheta_set, Sens_set, dSens_dTheta_set = funcs_pde.use_NN_set(Inputs_do,NN_set)[:4]

            #Compute Residual
            #dydt + y*dydx = 0
            ResPDE = Sens_set[0][1,:] + Outputs_set[0]*Sens_set[0][0,:]

            #Residuals gradient
            dResPDE_dTheta = dSens_dTheta_set[0][:,1,:] + Outputs_set[0]*dSens_dTheta_set[0][:,0,:] + dOutputs_dTheta_set[0][:,:]*Sens_set[0][0,:]

            #Returns
            return ResPDE, dResPDE_dTheta    

        ###########################################

        def analytical_sol(Inputs):

            # Analytical solution
            Outputs = np.zeros(Inputs.shape[1])
            
            for ii in range(Inputs.shape[1]):

                x = Inputs[0,ii]
                t = Inputs[1,ii]

                # Compute break points
                xb1 = 0.3 + 0.1*t
                xb2 = 0.4 + 0.2*t
                xb3 = 0.6 + 0.2*t
                xb4 = 0.7 + 0.1*t

                if x <= xb1:
                    Outputs[ii] = 0.1
                elif x > xb1 and x <= xb2:
                    Outputs[ii] = 0.1 + 0.1*(x-xb1)/(xb2-xb1)
                elif x > xb2 and x <= xb3:
                    Outputs[ii] = 0.2
                elif x > xb3 and x <= xb4:
                    Outputs[ii] = 0.2 - 0.1*(x-xb3)/(xb4-xb3)
                elif x > xb4:
                    Outputs[ii] = 0.1

            return Outputs

        ###########################################

        def compare_plot(NN_set):

            # Select y coordinates for the slices
            y_slices = [y0, yf]

            # Generate pair of slicing points
            Xpair_list = [[[x0, yy], [xf, yy]] for yy in y_slices]

            # The number of points will be equal to the training points
            n_list = [nx]*len(y_slices)

            # Titles
            title_list = [r'$t=%g$'%yy for yy in y_slices]

            # Call the slice function
            fig = plot_slice(NN_set,
                             analytical_sol,
                             Xpair_list,
                             n_list,
                             title_list,
                             x_label = r'$x$',
                             p0 = x0,
                             pf = xf)

            return [fig]

        # Levels for the contour plot
        level_min = 0.095
        level_max = 0.205

        ###########################################

        # Optimization parameters
        reg_factor = 0.0
        rhoKS = 0.5
        opt_options = {}

        if optimizer == 'ALM':

            opt_options = {'grad_opt':'bfgs',
                           'display':True,
                           'major_iterations':10,
                           'gamma':1.,
                           'delta_x_tol': 1e-8,
                           'minor_iterations':3000,
                           'grad_tol':1e-8,
                           'save_minor_hist':False,
                           'forced_iterations':0}

        elif optimizer == 'slsqp':
        
            opt_options = {'ftol':1e-9,
                           'disp':True,
                           'iprint':2,
                           'maxiter':5000}

    ###########################################
    ###########################################

    elif case_name == 'potflow_doublenet':

        ## USER INPUTS

        # Bounds
        r0 = 0.05
        ri = 0.15
        rf = 1.0

        # Refinement
        nr = 30
        ntheta = 40

        x0 = -rf
        y0 = -rf
        xf = rf
        yf = rf
        nx = 2*nr
        ny = 2*nr

        # BOUNDARY CONDITIONS

        # List of angles
        theta = np.linspace(0, 2*np.pi, ntheta+1)[:-1]

        # dudr=0 for inner face
        Inputs_inner = np.vstack([r0*np.cos(theta), r0*np.sin(theta)])
        Norms_inner = np.vstack([np.cos(theta), np.sin(theta)])
        Targets_dr_inner = 0.0*np.ones(ntheta)

        # Interface ring
        Inputs_interface = np.vstack([ri*np.cos(theta), ri*np.sin(theta)])
        
        # u=0 for outer face
        Inputs_outer = np.vstack([rf*np.cos(theta), rf*np.sin(theta)])
        Targets_outer = rf*np.cos(theta)
        
        ## DOMAIN
        
        if dist_type == 'structured':
        
            r_do = np.geomspace(r0,ri,nr)
            R_do, Theta_do = np.meshgrid(r_do,theta)
            R_do = R_do.ravel()
            Theta_do = Theta_do.ravel()
            Inputs_near = np.vstack([R_do*np.cos(Theta_do), R_do*np.sin(Theta_do)])

            r_do = np.linspace(ri,rf,nr)
            R_do, Theta_do = np.meshgrid(r_do,theta)
            R_do = R_do.ravel()
            Theta_do = Theta_do.ravel()
            Inputs_far = np.vstack([R_do*np.cos(Theta_do), R_do*np.sin(Theta_do)])
        
        elif dist_type == 'unstructured':

            # Near field
            r_rand = (ri-r0)*rng.random(nr*ntheta) + r0
            theta_rand = 2*np.pi*rng.random(nr*ntheta)
            Inputs_near = np.vstack([r_rand*np.cos(theta_rand), r_rand*np.sin(theta_rand)])
        
            # Far field
            r_rand = (rf-ri)*rng.random(nr*ntheta) + ri
            theta_rand = 2*np.pi*rng.random(nr*ntheta)
            Inputs_far = np.vstack([r_rand*np.cos(theta_rand), r_rand*np.sin(theta_rand)])

        ## NEURAL NETWORK

        # Create two neural networks for the field variable
        # One for near field and another for far field
        if num_neurons is None:
            num_neurons = [8, 8]
        NN_near = NN(num_inputs=2,
                     num_outputs=1,
                     num_neurons_list=num_neurons,
                     layer_type=['sigmoid']*len(num_neurons),
                     Theta_flat=123,
                     lower_bounds=[-ri,-ri],
                     upper_bounds=[ri,ri])
        NN_far = NN(num_inputs=2,
                    num_outputs=1,
                    num_neurons_list=num_neurons,
                    layer_type=['sigmoid']*len(num_neurons),
                    Theta_flat=123,
                    lower_bounds=[-rf,-rf],
                    upper_bounds=[rf,rf])
        NN_set = [NN_near, NN_far]

        ## RESIDUAL FUNCTIONS

        def Residual_BC(NN_set):
            # This function computes the residuals at the given points for multiple networks.
            # It also gives the residuals gradients with respect to the weights

            # For this problem I'll use
            # NN_set[0]: NN_u near
            # NN_set[1]: NN_u far

            # INNER BC
            num_innerbc_pts = len(Inputs_inner[0,:])

            #Use nearfield ANN on cylinder
            dSens_dTheta_inner, bla, ble, Sens_inner = NN_set[0].backpropagation(Inputs_inner)[1:5]

            # Add freestream contribution
            Sens_inner = Sens_inner + np.vstack([np.ones(num_innerbc_pts), np.zeros(num_innerbc_pts)])

            #Compute the total residual
            ResBC_inner = np.sum(Sens_inner*Norms_inner,axis=0) - Targets_dr_inner

            #Compute the MSE derivative
            dResBC_dTheta_inner = dSens_dTheta_inner[:,0,:]*Norms_inner[0,:] + dSens_dTheta_inner[:,1,:]*Norms_inner[1,:]

            #Add zeros to other weight derivatives
            dResBC_dTheta_inner = np.vstack([dResBC_dTheta_inner, np.zeros((len(NN_set[1].Theta_flat),Inputs_inner.shape[1]))])

            # OUTER BC
            #Use farfield ANN on outer BC points
            dOutputs_dTheta_outer, bla, ble, Outputs_outer = NN_set[1].backpropagation(Inputs_outer)[0:4]
            # Add freestream component
            Outputs_outer = Outputs_outer + Inputs_outer[0,:]
            #Compute the total residual
            ResBC_outer = Outputs_outer-Targets_outer
            #Compute the MSE derivative
            dResBC_dTheta_outer = dOutputs_dTheta_outer
            #Add zeros to other weight derivatives
            dResBC_dTheta_outer = np.vstack([np.zeros((len(NN_set[0].Theta_flat),Inputs_outer.shape[1])),dResBC_dTheta_outer])

            # INTERFACE
            #Use farfield ANN on interface
            dOutputs_dTheta_outer, dSens_dTheta_outer, ble, Outputs_outer, Sens_outer = NN_set[1].backpropagation(Inputs_interface)[0:5]
            #Use nearfield ANN on interface
            dOutputs_dTheta_inner, dSens_dTheta_inner, ble, Outputs_inner, Sens_inner = NN_set[0].backpropagation(Inputs_interface)[0:5]
            # Compute residual at interface
            ResBC_int = np.hstack([Outputs_outer - Outputs_inner,
                                   Sens_outer[0,:] - Sens_inner[0,:],
                                   Sens_outer[1,:] - Sens_inner[1,:]])
            dResBC_dTheta_int = np.hstack([np.vstack([-dOutputs_dTheta_inner,dOutputs_dTheta_outer]),
                                           np.vstack([-dSens_dTheta_inner[:,0,:],dSens_dTheta_outer[:,0,:]]),
                                           np.vstack([-dSens_dTheta_inner[:,1,:],dSens_dTheta_outer[:,1,:]])])

            # Join everything
            ResBC = np.hstack([ResBC_outer, ResBC_inner, ResBC_int])
            dResBC_dTheta = np.hstack([dResBC_dTheta_outer, dResBC_dTheta_inner, dResBC_dTheta_int])

            #RETURNS
            return ResBC, dResBC_dTheta

        ###########################################

        def Residual_PDE(NN_set):

            #First variable is x, second is y (x1=x, x2=y)
            #For this problem I'll use
            #NN_set: [NN_near, NN_far]
            
            # Obtain derivatives
            dHess_dTheta_near, bla, ble, Hess_near = NN_set[0].backpropagation(Inputs_near)[2:6]
            dHess_dTheta_far, bla, ble, Hess_far = NN_set[1].backpropagation(Inputs_far)[2:6]
            
            #Compute Residual
            #d2Tdx2 + d2Tdy2 = 0
            ResPDE_near = Hess_near[0,0,:] + Hess_near[1,1,:]
            ResPDE_far = Hess_far[0,0,:] + Hess_far[1,1,:]
            
            ResPDE = np.hstack([ResPDE_near, ResPDE_far])
            
            #Residuals gradient
            dResPDE_dTheta_near = dHess_dTheta_near[:,0,0,:] + dHess_dTheta_near[:,1,1,:]
            dResPDE_dTheta_far = dHess_dTheta_far[:,0,0,:] + dHess_dTheta_far[:,1,1,:]
            
            # Add zeroes for other derivatives correponding to the weights of the
            # other ANNs
            # [[dResPDE_dTheta_near, ZEROES],
            #  [ZEROES, dResPDE_dTheta_far]]
            size_near = dResPDE_dTheta_near.shape
            size_far = dResPDE_dTheta_far.shape
            
            dResPDE_dTheta = np.vstack( [ np.hstack([dResPDE_dTheta_near,  np.zeros((size_near[0], size_far[1])) ]),        
                                          np.hstack([np.zeros((size_far[0], size_near[1])), dResPDE_dTheta_far])  ])
            
            #Returns
            return ResPDE, dResPDE_dTheta

        ###########################################

        # ANALYTICAL SOLUTION

        def analytical_sol(Inputs):

            # Get coordinates
            x = Inputs[0,:]
            y = Inputs[1,:]

            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y,x)

            # Initialize Output array
            Outputs = (r0**2/r)*np.cos(theta)

            return Outputs

        ###########################################

        def compare_plot(NN_set):

            # Get NNs again
            NN = NN_set[0]
            
            # Let's specify the BC conditions
            theta = np.linspace(0, 2*np.pi, ntheta+1)[:-1]
            Inputs_inner = np.vstack([r0*np.cos(theta), r0*np.sin(theta)])
            
            # PLOT
            Sens  = NN.feedforward(Inputs_inner)[1]
            u = Sens[0,:] + 1.0
            v = Sens[1,:]
            cp = 1 - (u**2+v**2)
            cp_an = 2*np.cos(2*theta)-1
            
            # Plotting results with sensitivities
            # More imports
            fig = plt.figure()
            #plt.subplot(211)
            plt.plot(Inputs_inner[0,:],cp,'o',label='ANN')
            plt.plot(Inputs_inner[0,:],cp_an,'k',label='Analytical')
            plt.xlabel(r'$x$',fontsize=18)
            plt.ylabel(r'$Cp$',fontsize=18)
            plt.legend(loc='best',fontsize=18)

            fig.savefig('./potflow_doublenet/potflow_doublenet_compare_plot.pdf',dpi=300)
            plt.close(fig)
            return [fig]

        def plot_streamlines(NN_set):
                import numpy as np
                import matplotlib.pyplot as plt

                import colormaps as cmaps
                #Get NNs again
                NN_near = NN_set[0]
                NN_far = NN_set[1]

                #PLOT NEAR FIELD
                c = np.cos(45*np.pi/180)
                x_near = np.linspace(-0.1, 0.1, 101)
                y_near = np.linspace(-0.1, 0.1, 101)
                X_near, Y_near = np.meshgrid(x_near, y_near)
                # assemble x-y array
              
                Inputs = np.vstack([X_near.ravel(),Y_near.ravel()])
                
                # run ANN
                Sens  = NN_near.feedforward(Inputs)[1]
                u = Sens[0,:] + 1.0
                v = Sens[1,:]
                
                for i in range(len(X_near.ravel())):
                    if (np.sqrt(X_near.ravel()[i]**2+Y_near.ravel()[i]**2) < 0.05):
                        u[i] = 0
                        v[i] = 0
                # reshape result to mesh shape
                U_near = u.reshape(X_near.shape)
                V_near = v.reshape(X_near.shape)

                #Plotting results with sensitivities
                #More imports
                from mpl_toolkits.mplot3d import Axes3D
                from matplotlib import cm
                from matplotlib.ticker import LinearLocator, FormatStrFormatter
                import matplotlib.pyplot as plt
                import numpy as np

                import colormaps as cmaps
                plt.register_cmap(name='viridis', cmap=cmaps.viridis)
                plt.set_cmap(cmaps.viridis)

                fig, ax = plt.subplots(figsize=(8,7))
                ax.streamplot(x_near, y_near, U_near, V_near, density=1.2)
                plt.plot(Inputs_near[0,:],Inputs_near[1,:],'o',markersize=3,alpha=0.3,c='r')
                plt.plot(Inputs_far[0,:],Inputs_far[1,:],'o',markersize=3,alpha=0.3,c='r')
                plt.plot(Inputs_interface[0,:],Inputs_interface[1,:],'o',markersize=3,alpha=0.3,c='k')
                circle = plt.Circle((0.0, 0.0), r0, color="white", alpha=0.9, ec='k', zorder=3)
                ax.add_patch(circle)

                plt.xlim([-0.1, 0.1])
                plt.ylim([-0.1, 0.1])
                #plt.axis('equal')

                # Add labels
                plt.xlabel(r'$x$',fontsize=18)
                plt.ylabel(r'$y$',fontsize=18)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)

                # Save figure
                fig.savefig('./potflow_doublenet/potflow_doublenet_streamlines.pdf',dpi=300)
                plt.close(fig)

                # RETURNS
                return fig #The user should return the figure handle

        # Levels for the contour plot
        level_min = -1.05
        level_max = 1.05

        ###########################################


        # Optimization parameters
        reg_factor = 0.001
        rhoKS = 1.5
        opt_options = {}

        if optimizer == 'ALM':

            opt_options = {'grad_opt':'bfgs',
                           'display':True,
                           'major_iterations':10,
                           'gamma':1.,
                           'delta_x_tol':1e-6,
                           'minor_iterations':5000,
                           'grad_tol':1e-6,
                           'save_minor_hist':False}

        elif optimizer == 'slsqp':
        
            opt_options = {'ftol':1e-9,
                           'disp':True,
                           'iprint':2,
                           'maxiter':10000}

        # DUMMY
        Inputs_do = np.hstack([Inputs_near,Inputs_far])
        Inputs_bc = Inputs_outer
        Targets_bc = Targets_outer - Inputs_outer[0,:]

    ###########################################
    ###########################################

    elif case_name == 'potflow_singlenet':

        ## USER INPUTS

        # Bounds
        r0 = 0.05
        ri = 0.15
        rf = 1.0

        # Refinement
        nr = 30
        ntheta = 40 # Increase this if result is not symmetric (at least 30)

        x0 = -rf
        y0 = -rf
        xf = rf
        yf = rf
        nx = 2*nr
        ny = 2*nr

        # BOUNDARY CONDITIONS

        # List of angles
        theta = np.linspace(0, 2*np.pi, ntheta+1)[:-1]

        # dudr=0 for inner face
        Inputs_inner = np.vstack([r0*np.cos(theta), r0*np.sin(theta)])
        Norms_inner = np.vstack([np.cos(theta), np.sin(theta)])
        Targets_dr_inner = 0.0*np.ones(ntheta)

        ## Interface ring
        #Inputs_interface = np.vstack([ri*np.cos(theta), ri*np.sin(theta)])
        
        # u=0 for outer face
        Inputs_outer = np.vstack([rf*np.cos(theta), rf*np.sin(theta)])
        Targets_outer = rf*np.cos(theta)
        
        ## DOMAIN
        
        if dist_type == 'structured':
        
            r_do_near = np.geomspace(r0,ri,nr)
            R_do_near, Theta_do_near = np.meshgrid(r_do_near,theta)
            R_do_near = R_do_near.ravel()
            Theta_do_near = Theta_do_near.ravel()
            #Inputs_near = np.vstack([R_do*np.cos(Theta_do), R_do*np.sin(Theta_do)])

            r_do_far = np.linspace(ri,rf,nr)
            R_do_far, Theta_do_far = np.meshgrid(r_do_far,theta)
            R_do_far = R_do_far.ravel()
            Theta_do_far = Theta_do_far.ravel()
            #Inputs_far = np.vstack([R_do*np.cos(Theta_do), R_do*np.sin(Theta_do)])
            Inputs_do = np.hstack([np.vstack([R_do_near*np.cos(Theta_do_near),R_do_near*np.sin(Theta_do_near)]),np.vstack([R_do_far*np.cos(Theta_do_far),R_do_far*np.sin(Theta_do_far)])])
            
        elif dist_type == 'unstructured':

            # Near field
            r_rand_near = (ri-r0)*rng.random(nr*ntheta) + r0
            theta_rand_near = 2*np.pi*rng.random(nr*ntheta)
            #Inputs_near = np.vstack([r_rand*np.cos(theta_rand), r_rand*np.sin(theta_rand)])
        
            # Far field
            r_rand_far = (rf-ri)*rng.random(nr*ntheta) + ri
            theta_rand_far = 2*np.pi*rng.random(nr*ntheta)
            #Inputs_far = np.vstack([r_rand*np.cos(theta_rand), r_rand*np.sin(theta_rand)])
            Inputs_do = np.hstack([np.vstack([r_rand_near*np.cos(theta_rand_near), r_rand_near*np.sin(theta_rand_near)]),np.vstack([r_rand_far*np.cos(theta_rand_far), r_rand_far*np.sin(theta_rand_far)])])

        ## NEURAL NETWORK
        # Create one neural networks for the field variable
        if num_neurons is None:
            num_neurons = [12, 12]
        NN_u = NN(num_inputs=2,
                  num_outputs=1,
                  num_neurons_list=num_neurons,
                  layer_type=['sigmoid']*len(num_neurons),
                  Theta_flat=123,
                  lower_bounds=[-rf,-rf],
                  upper_bounds=[rf,rf])
        NN_set = [NN_u]

        ## RESIDUAL FUNCTIONS

        def Residual_BC(NN_set):
            # This function computes the residuals at the given points for multiple networks.
            # It also gives the residuals gradients with respect to the weights

            # For this problem I'll use
            # NN_set[0]: NN_u

            # OUTER BC

            # Use all NNs (even though we have only one for this example)
            Outputs_set, dOutputs_dTheta_set = funcs_pde.use_NN_set(Inputs_outer,NN_set)[0:2]
            
            # Compute the total residual
            ResBC_outer = Outputs_set[0]

            # Compute the MSE derivative
            dResBC_dTheta_outer = dOutputs_dTheta_set[0]
            
            # INNER BC
            
            # Use all NNs (even though we have only one for this example)
            Sens_set, dSens_dTheta_set = funcs_pde.use_NN_set(Inputs_inner,NN_set)[2:4]
            
            # Get derivatives of the first ANN
            dudx = Sens_set[0][0,:]
            dudx_dTheta = dSens_dTheta_set[0][:,0,:]
            dudy = Sens_set[0][1,:]
            dudy_dTheta = dSens_dTheta_set[0][:,1,:]

            # Compute BC residuals
            ResBC_inner = (dudx + 1.0)*Inputs_inner[0,:] + (dudy)*Inputs_inner[1,:]

            # Compute the residual derivative
            dResBC_dTheta_inner = (dudx_dTheta)*Inputs_inner[0,:] + (dudy_dTheta)*Inputs_inner[1,:]
            
            # Join everything
            pen = 10**0
            ResBC = np.hstack([ResBC_inner*pen, ResBC_outer])
            dResBC_dTheta = np.hstack([dResBC_dTheta_inner*pen, dResBC_dTheta_outer])
            
            #RETURNS
            return ResBC, dResBC_dTheta

        ###########################################

        def Residual_PDE(NN_set):

            #First variable is x, second is y (x1=x, x2=y)
            #For this problem I'll use
            #NN_set: [NN_near, NN_far]
            
            #Use all NNs
            Hess_set, dHess_dTheta_set = funcs_pde.use_NN_set(Inputs_do,NN_set)[4:]

            #Compute Residual
            #d2udx2 + d2udy2 = 0
            ResPDE = Hess_set[0][0,0,:] + Hess_set[0][1,1,:]

            #Residuals gradient
            dResPDE_dTheta = dHess_dTheta_set[0][:,0,0,:] + dHess_dTheta_set[0][:,1,1,:]
            
            #Returns
            return ResPDE, dResPDE_dTheta

        ###########################################

        # ANALYTICAL SOLUTION

        def analytical_sol(Inputs):

            # Get coordinates
            x = Inputs[0,:]
            y = Inputs[1,:]

            # Initialize Output array
            Outputs = r0**2*x/(x**2+y**2)

            return Outputs

        ###########################################
        def plot_streamlines(NN_set):
            import numpy as np
            import matplotlib.pyplot as plt

            import colormaps as cmaps

            #Get NNs again
            NN = NN_set[0]

            #PLOT NEAR FIELD
            c = np.cos(45*np.pi/180)
            x_near = np.linspace(-0.1, 0.1, 101)
            y_near = np.linspace(-0.1, 0.1, 101)
            X_near, Y_near = np.meshgrid(x_near, y_near)
            # assemble x-y array
            Inputs = np.vstack([X_near.ravel(), Y_near.ravel()])
            # run ANN
            Sens  = NN.feedforward(Inputs)[1]
            u = Sens[0,:] + 1.0
            v = Sens[1,:]
            
            for i in range(len(X_near.ravel())):
                if (np.sqrt(X_near.ravel()[i]**2+Y_near.ravel()[i]**2) < 0.05):
                    u[i] = 0
                    v[i] = 0
                    
            # reshape result to mesh shape
            U_near = u.reshape(X_near.shape)
            V_near = v.reshape(X_near.shape)

            #Plotting results with sensitivities
            #More imports
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib import cm
            from matplotlib.ticker import LinearLocator, FormatStrFormatter
            import matplotlib.pyplot as plt
            import numpy as np

            import colormaps as cmaps
            plt.register_cmap(name='viridis', cmap=cmaps.viridis)
            plt.set_cmap(cmaps.viridis)

            fig, ax = plt.subplots(figsize=(8,7))
            ax.streamplot(x_near, y_near, U_near, V_near, density=1.2)
            plt.plot(Inputs_do[0,:],Inputs_do[1,:],'o',markersize=3,alpha=0.3,c='r')
            circle = plt.Circle((0.0, 0.0), r0, color="white", alpha=0.9, ec='k', zorder=3)
            ax.add_patch(circle)

            plt.xlim([-0.1, 0.1])
            plt.ylim([-0.1, 0.1])
            #plt.axis('equal')

            # Add labels
            plt.xlabel(r'$x$',fontsize=18)
            plt.ylabel(r'$y$',fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            # Save figure
            fig.savefig('./potflow_singlenet/potflow_singlenet_streamlines.pdf',dpi=300)
            plt.close(fig)

            # RETURNS
            return fig #The user should return the figure handle

        def compare_plot(NN_set):

            # Get NNs again
            NN = NN_set[-1]
            
            # PLOT
            Sens = NN.feedforward(Inputs_inner)[1]
            u = Sens[0,:] + 1.0
            v = Sens[1,:]
            cp = 1 - (u**2+v**2)
            theta = np.linspace(0, np.pi, 101)[:-1]
            xx = r0*np.cos(theta)
            cp_an = 2*np.cos(2*theta)-1

            
            # Plotting results with sensitivities
            # More imports
            fig = plt.figure()

            #plt.subplot(211)

            plt.plot(Inputs_inner[0,:],cp,'o',label='ANN')
            plt.plot(xx,cp_an,'k',label='Analytical')
            plt.xlabel(r'$x$',fontsize=18)
            plt.ylabel(r'$Cp$',fontsize=18)
            plt.legend(loc='best',fontsize=18)
            
            fig.savefig('./potflow_singlenet/potflow_singlenet_compare_plot.pdf',dpi=300)
            plt.close(fig)
            return [fig]

        # Levels for the contour plot
        level_min = -1.05
        level_max = 1.05

        ###########################################

        # Optimization parameters
        reg_factor = 0.001
        rhoKS = 1.5
        opt_options = {}

        if optimizer == 'ALM':

            opt_options = {'grad_opt':'bfgs',
                           'display':True,
                           'major_iterations':10,
                           'gamma':1.,
                           'delta_x_tol':1e-6,
                           'minor_iterations':5000,
                           'grad_tol':1e-6,
                           'save_minor_hist':False}

        elif optimizer == 'slsqp':
        
            opt_options = {'ftol':1e-12,
                           'disp':True,
                           'iprint':2,
                           'maxiter':50000}

        # DUMMY
        Inputs_bc  = Inputs_outer
        Targets_bc = Inputs_outer[0,:]*0.0

    #==========================================

    # Define files and directories
    hist_file = case_name+'/results.pickle'
    image_dir = './'+case_name

    # Check if the images directory exists
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Check if we will wipe out everything inside the output directory
    if clean_image_dir:
        os.system('rm ' + image_dir+'/*.png')
        os.system('rm ' + image_dir+'/*.pdf')

    # Define common plot_function for runtime verification
    def plot_function(NN_set):
        
        # Use helper function
        fig_list1 = plot_surface(NN_set,
                                  x0, xf, y0, yf, nx, ny,
                                  Inputs_bc, Targets_bc,Inputs_do,
                                  level_min, level_max,axis_label)

        # Generate particular plot of the case
        fig_list = compare_plot(NN_set)
        if case_name == 'potflow_doublenet' or case_name == 'potflow_singlenet':
            fig = plot_streamlines(NN_set)
        # RETURNS
        return fig_list1 + fig_list # The user should return a list of figure handles

    # Define common function to compute validation metric
    def val_function(NN_set):
        nxn = 301
        if case_name == 'potflow_singlenet':
            #Get NNs again
            NN_u = NN_set[-1]
            ntheta = nxn
            nr     = nxn
            theta = np.linspace(0, 2*np.pi, ntheta+1)[:-1]
            r_do = np.linspace(r0,rf,nr)
            R_do, Theta_do = np.meshgrid(r_do,theta)
            R_do = R_do.ravel()
            Theta_do = Theta_do.ravel()
            Inputs = np.vstack([R_do*np.cos(Theta_do), R_do*np.sin(Theta_do)])
            
            # Evaluate using the ANN
            Outputs_T = NN_u.feedforward(Inputs)[0]
            
            # Analytical solution
            Outputs = analytical_sol(Inputs)

            # Compute MSE over the validation set
            valMSE = np.sum((Outputs_T - Outputs)**2)/2/len(Outputs)
        
        elif case_name == 'potflow_doublenet':
           #Get NNs again
            NN_near = NN_set[0]
            NN_far  = NN_set[1]

            ntheta = nxn
            nr     = nxn
            theta = np.linspace(0, 2*np.pi, ntheta+1)[:-1]
            r_do = np.linspace(r0,rf,nr)
            R_do, Theta_do = np.meshgrid(r_do,theta)
            R_do = R_do.ravel()
            Theta_do = Theta_do.ravel()

            #Inputs_near and Far
            x_near = []
            y_near = []
            x_far  = []
            y_far  = []
            for i in range(len(R_do)):
                if (R_do[i] <= ri):
                    x_near.append(R_do[i]*np.cos(Theta_do[i]))
                    y_near.append(R_do[i]*np.sin(Theta_do[i]))
                else:
                    x_far.append(R_do[i]*np.cos(Theta_do[i]))
                    y_far.append(R_do[i]*np.sin(Theta_do[i]))

            Inputs_near = np.vstack([x_near, y_near])
            Inputs_far  = np.vstack([x_far, y_far])
            # Evaluate using the ANN
            Outputs_T_near = NN_near.feedforward(Inputs_near)[0]
            Outputs_T_far  = NN_far.feedforward(Inputs_far)[0]

            # Analytical solution
            Outputs_near = analytical_sol(Inputs_near)
            Outputs_far  = analytical_sol(Inputs_far)

            # Compute MSE over the validation set
            valMSE = (np.sum((Outputs_T_near - Outputs_near)**2)+np.sum((Outputs_T_far - Outputs_far)**2))/2/(len(Outputs_near)+len(Outputs_far))
        else:
            #Get NNs again
            NN_u = NN_set[-1]

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

    def MSEhist_function(NN_set,Theta_flat_hist,Theta_hist_ALM,interval):
        import matplotlib.pyplot as plt
        
        num_PDE     = len(Residual_PDE(NN_set)[0])
        num_BC      = len(Residual_BC(NN_set)[0])
        num_weights = len(Theta_flat_hist[0])
        
        '''
        #Iterations history
        valMSE_array       = np.zeros(len(Theta_flat_hist))
        Residual_PDE_array = np.zeros(len(Theta_flat_hist))
        Residual_BC_array  = np.zeros(len(Theta_flat_hist))
        weights_array      = np.zeros(len(Theta_flat_hist))
        iteration_array    = np.linspace(1,len(Theta_flat_hist),len(Theta_flat_hist))*interval
        
        
        for i in range(len(Theta_flat_hist)):
            print('iteration = ',i)
            funcs_pde.reassign_theta(NN_set,Theta_flat_hist[i])
            valMSE_array[i]        = val_function(NN_set)
            Residual_PDE_array[i]  = np.sum(Residual_PDE(NN_set)[0]**2)/2/num_PDE
            Residual_BC_array[i]   = np.sum(Residual_BC(NN_set)[0]**2)/2/num_BC
            weights_array[i]       = np.sum(Theta_flat_hist[i]**2)/2/num_weights
            iteration_array[i]     = int(iteration_array[i])
            
        ## MSE evolution over optimizer iterations
        fig = plt.figure()
        plt.semilogy(iteration_array,valMSE_array, linewidth=2, label=r'$MSE_{val}$')
        plt.semilogy(iteration_array,Residual_PDE_array, linewidth=2, label=r'PDE residuals')
        plt.semilogy(iteration_array,Residual_BC_array, linewidth=2, label=r'BC residuals')
        plt.semilogy(iteration_array,weights_array, linewidth=2, label=r'ANN weights')
        plt.legend(loc='best',fontsize=18)
        
        # Add labels
        plt.xlabel(r'$Iterations$',fontsize=18)
        plt.ylabel(r'$Mean\:\:Square$',fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        # Save figure
        fig.savefig(case_name + '_MSE_iterations.pdf',dpi=300)
        '''
        
        # Transpose matrix to facilitate for loop
        # Now each weight iteration is a line
        Theta_hist_ALM = Theta_hist_ALM.T

        #ALM history
        valMSE_ALM         = np.zeros(len(Theta_hist_ALM))
        Residual_PDE_ALM   = np.zeros(len(Theta_hist_ALM))
        Residual_BC_ALM    = np.zeros(len(Theta_hist_ALM))
        weights_ALM        = np.zeros(len(Theta_hist_ALM))
        iterations_ALM     = np.linspace(0,len(Theta_hist_ALM)-1,len(Theta_hist_ALM))
        
        for j in range(len(Theta_hist_ALM)):
            funcs_pde.reassign_theta(NN_set,Theta_hist_ALM[j])
            valMSE_ALM[j]        = val_function(NN_set)
            print('MSEval at the '+ str(j) + '-th ALM iteration = ' + str(valMSE_ALM[j]))
            Residual_PDE_ALM[j]  = np.sum(Residual_PDE(NN_set)[0]**2)/2/num_PDE
            Residual_BC_ALM[j]   = np.sum(Residual_BC(NN_set)[0]**2)/2/num_BC
            weights_ALM[j]       = np.sum(Theta_hist_ALM[j]**2)/2/num_weights
            iterations_ALM[j]    = int(iterations_ALM[j])
             
        ## MSE evolution over ALM iterations
        fig2 = plt.figure()
        plt.semilogy(iterations_ALM,valMSE_ALM, '-o', linewidth=2, label=r'$MSE_{val}$')
        plt.semilogy(iterations_ALM,Residual_PDE_ALM, '-o', linewidth=2, label=r'PDE residuals')
        plt.semilogy(iterations_ALM,Residual_BC_ALM, '-o', linewidth=2, label=r'BC residuals')
        plt.semilogy(iterations_ALM,weights_ALM, '-o', linewidth=2, label=r'ANN weights')
        
        # Comment if you are generating the plot for the article
        # with the MSE history for the lin_adv case
        plt.legend(loc='best',fontsize=18)

        # Uncomment just for the linear advection case
        #colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        #plt.text(7, 1e-4, r'$\bm{MSE_{val}}$', color=colors[0], fontsize=16)
        #plt.text(7, 1e-12, r'$\bm{\mathrm{PDE \; residuals}}$', color=colors[1], fontsize=16)
        #plt.text(5, 0.5e-16, r'$\bm{\mathrm{BC \; residuals}}$', color=colors[2], fontsize=16)
        #plt.text(7, 1e0, r'$\bm{\mathrm{ANN \; weights}}$', color=colors[3], fontsize=16)
        
        # Add labels
        plt.xlabel(r'$\mathrm{ALM \; iterations}$',fontsize=18)
        plt.ylabel(r'$\mathrm{Mean \; Squared \; Value}$',fontsize=18)
        plt.xticks(iterations_ALM,fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        # Save figure
        fig2.savefig('./' + case_name + '/' + case_name + '_MSE_ALM.pdf',dpi=300)
        plt.close(fig2)

    # Return variables of the corresponding case
    return [NN_set, Residual_BC, Residual_PDE, plot_function, val_function, MSEhist_function,
            reg_factor, rhoKS, opt_options, hist_file, image_dir,
            Inputs_do, Inputs_bc, Targets_bc]

###########################################
# AUXILIARY FUNCTIONS
###########################################

def create_2Ddomain(x0, xf, nx,
                    y0, yf, ny,
                    dist='structured',
                    no_borders=False,
                    rng=None):
    '''
    This function distributed nx*ny points over a 2D domain in
    structured or unstructured fashion.

    Inputs_do also contains all border points.

    INPUTS
    x0: float -> Lower bound of the domain's first dimension
    xf: float -> Upper bound of the domain's first dimension
    nx: integer -> Number of nodes along the first dimension
    y0: float -> Lower bound of the domain's second dimension
    yf: float -> Upper bound of the domain's second dimension
    ny: integer -> Number of nodes along the second dimension
    dist: string -> Either 'structured' or 'unstructured'
    no_borders: logical -> Eliminate the boundary points from
                           the domain (Inputs_do)
    rng: Random number generator -> Generator given by np.random.default_rng.
                                    Can be a generator already defined before.
    '''

    # Check if used defined random number generator
    if rng is None:
        rng = np.random.default_rng()

    ## BORDERS

    x_do = np.linspace(x0,xf,nx)
    y_do = np.linspace(y0,yf,ny)

    Inputs_lower = np.vstack([x_do, y0*np.ones(nx)]) #y=0
        
    Inputs_left = np.vstack([x0*np.ones(ny-1), y_do[1:]]) #x=0
           
    Inputs_top = np.vstack([x_do[1:-1], yf*np.ones(nx-2)]) #y=1
    
    Inputs_right = np.vstack([xf*np.ones(ny-1), y_do[1:]]) #x=1
    
    ## INTERIOR POINTS

    if dist == 'structured':

        X_do, Y_do = np.meshgrid(x_do[1:-1],y_do[1:-1])
        Inputs_do = np.vstack([X_do.ravel(),Y_do.ravel()])

    elif dist == 'unstructured':
    
        n_samples = (nx-2)*(ny-2)

        x_do = (xf-x0)*rng.random(n_samples) + x0
        y_do = (yf-y0)*rng.random(n_samples) + y0

        Inputs_do = np.vstack([x_do,y_do])

    # Now assemble everything
    if not no_borders:
        Inputs_do = np.hstack([Inputs_do,Inputs_lower,Inputs_right,Inputs_top,Inputs_left]) # Make sure we check PDE on BC points

    return Inputs_do, Inputs_lower, Inputs_right, Inputs_top, Inputs_left

###########################################

def plot_surface(NN_set,
                 x0, xf, y0, yf, nx, ny,
                 Inputs_bc, Targets_bc, Inputs_do,
                 level_min=-0.05, level_max=1.05,axis_label=['x','y']):
    
    '''
    This function plots a surface plot for the given ANN and BCs

    level_min and level_max are the bounds for the contour plot.
    '''

    #Get NNs again
    NN_T = NN_set[-1]

    #PLOT
    x = np.linspace(x0,xf,nx*3)
    y = np.linspace(y0,yf,ny*3)
    X,Y = np.meshgrid(x,y)
    Inputs = np.vstack([X.ravel(),Y.ravel()])
    Outputs_T = NN_T.feedforward(Inputs)[0]
    Z_T = Outputs_T.reshape(X.shape)

    #Plotting results with sensitivities
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
    CS = plt.contourf(X, Y, Z_T, cmap=cm.viridis, levels=np.linspace(level_min,level_max,11))
    plt.plot(Inputs_do[0,:],Inputs_do[1,:],'o',markersize=3,c='r')
    plt.scatter(Inputs_bc[0,:],Inputs_bc[1,:],c='r')
    plt.xlabel(r'$' + axis_label[0] + '$',fontsize=18)
    plt.ylabel(r'$' + axis_label[1] + '$',fontsize=18)
    cb = plt.colorbar(CS)
    cb.ax.tick_params(labelsize=14)
    cb.set_label(label=r'$u$',size=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig2.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    plt.tight_layout()

    # RETURNS
    return [fig2] # The user should return the figure handle

###########################################

def plot_slice(NN_set,
               analytical_sol,
               Xpair_list,
               n_list,
               title_list,
               x_label = r'$\alpha$',
               figsize=(8,6),
               p0 = 0,
               pf = 1):
    '''
    This functions plots a figure containing
    several domain slices

    INPUTS:
    Xpair_list: list -> List of initial and final points of the slices
                        Xpair_list = [[[xs1, ys1], [xf1, yf1]],
                                      [[xs2, ys2], [xf2, yf2]],
                                      [[xs3, ys3], [xf3, yf3]],
                                      .........................]
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
        plt.plot(alpha_ref, Outputs_ref_an, 'k', label='Analytical', linewidth=2)

        # Plot ANN solution
        plt.scatter(alpha, Outputs, c='r', label='ANN training points')

        # Plot ANN solution at refined points
        plt.plot(alpha_ref, Outputs_ref, 'r', linewidth=2, label='ANN')

        # Add title to the plot
        plt.title(r'' + title, fontsize=18)
        plt.ylabel(r'$u$',fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # Add legend to the first plot only
        if slice_id == 1:
            plt.legend(loc='best',fontsize=18)

        # Increment slice counter
        slice_id = slice_id + 1
        fig.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places

    # Add label to the horizontal axis
    plt.xlabel(r''+ x_label,fontsize=18)
    # Adjust the plot
    plt.tight_layout()

    return fig

###########################################

def plot_canonical(NN_set, plot_function, image_name):
    '''
    This function takes the current state of the ANN and
    generates the plot of plot_function

    INPUTS:

    plot_function: function handle -> This function should receive
                   a NN_set list (list o NN objects) and return a list
                   of figure handles to be saved.
    '''
        
    figs = plot_function(NN_set)
    for ii,fig in enumerate(figs):
        if ii == 0:
            fig.savefig(image_name+'_domain',dpi=300)
            plt.close(fig)
        elif ii == 1:
            fig.savefig(image_name+'_slice',dpi=300)
            plt.close(fig)
        else:
            fig.savefig(image_name+'_'+str(ii),dpi=300)
            plt.close(fig)
