#GENERAL IMPORTS
from __future__ import division
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import time

###########################################

def train_pde_nn(NN_set, Residual_BC, Residual_PDE,
                 optimizer = 'ALM', opt_options = {},
                 reg_factor = 0.0, metric = 'MSE', rhoKS = 0.5, strategy = 3,
                 hist_file = None, load_hist = False,
                 plot_function = None, image_dir = '.', save_interval = 500,
                 check_gradients = False):
    '''     
    This function uses the backpropagation algorithm to train the ANN
    to represent the solution of a PDE.
    The backpropagation is performed with a wrapped Fortran code.
     
    INPUTS:
    NN_set: list of NN objects -> List containing Neural Networks used to simulate the PDE.
    Residual_BC: function -> This function should receive NN_set and return an array of
                             size m_BC containing the BC residuals at m_BC points.
                             It should also return another array of size p*m_BC,
                             containing the gradients of these residuals with respect to
                             the p weights of all NNs of NN_set.
    Residual_PDE: function -> This function should receive NN_set and return an array of
                              size m_PDE containing the PDE residuals at m_PDE points.
                              It should also return another array of size p*m_PDE,
                              containing the gradients of these residuals with respect to
                              the p weights of all NNs of NN_set.
    check_gradients: logical -> Flag to check gradients of the residual functions with
                                 finite differences.
    rhoKS: float -> KS function parameter.
    '''
    
    # Dummy result for the ALM case
    Theta_flat_ALM_hist = None

    # Check if we start a new optimization or use
    # previous information
    if not load_hist:

        # Get initial set of weights
        Theta_flat0 = [] # Initialize set of weights
        for index in range(len(NN_set)):
            Theta_flat0 = np.hstack([Theta_flat0, NN_set[index].Theta_flat])

        # Initialize weight history variable
        Theta_flat_hist = []

        # Initialize objective history
        obj_hist = []

        # Initialize function call counter.
        # We initialize it as a list so we can modify its contents
        # inside the objective function
        num_calls = [0]

    else: # Load previous history

        # Load final state from the pickle file
        # ATTENTION: We do not overwrite the ANN and optimizer options
        # using the pickle file information
        Theta_flat_hist, obj_hist, num_calls = load_history(hist_file)[1:4]

        # Get the initial point
        Theta_flat0 = Theta_flat_hist[-1]

    ###########################################

    # Define function to compute all residuals.
    # This function can be used either for Residual_PDE or Residual_BC
    def allRes(Theta_flat, Residual_func):

        # Reassign weights
        reassign_theta(NN_set,Theta_flat)

        # Get isolated residuals
        Res = Residual_func(NN_set)[0]
           
        # RETURNS
        return Res
    
    def allResGrad(Theta_flat, Residual_func):

        # Reassign weights
        reassign_theta(NN_set,Theta_flat)

        # Get isolated residuals
        Res, dRes_dTheta = Residual_func(NN_set)

        # RETURNS
        return Res, dRes_dTheta

    #--------------

    # Define function to compute averaged residuals.
    # This function can be used either for Residual_PDE or Residual_BC
    def avgRes(Theta_flat, Residual_func):

        # Reassign weights
        reassign_theta(NN_set,Theta_flat)

        # Get isolated residuals
        Res = Residual_func(NN_set)[0]

        # Compute averaged metric
        if metric == 'MSE':

            # Compute mean squared error of residuals
            avg = np.sum(Res**2)/2/len(Res)

        elif metric == 'KS':

            # We want to find the worst residual magnitude, so we square
            # the residuals for the KS constraint
            Res2 = Res**2

            # Find the worst residual
            Res2max = np.max(Res2)

            # Compute error KS function of residuals
            avg = Res2max + 1/rhoKS*np.log(np.sum(np.exp(rhoKS*(Res2 - Res2max))))
           
        # RETURNS
        return avg
    
    def avgResGrad(Theta_flat, Residual_func):

        # Reassign weights
        reassign_theta(NN_set,Theta_flat)

        # Get isolated residuals
        Res, dRes_dTheta = Residual_func(NN_set)

        # Compute derivative of the averaged metric
        if metric == 'MSE':

            # Compute mean squared error of residuals
            avg = sum(Res**2)/2/len(Res)

            # Compute derivative of the residuals MSE
            davg_dTheta = Res.dot(dRes_dTheta.T)/len(Res)

        elif metric == 'KS':

            # We want to find the worst residual magnitude, so we square
            # the residuals for the KS constraint
            Res2 = Res**2

            # Find the worst residual
            maxID = np.argmax(Res2)
            Res2max = Res2[maxID]
            Resmax = Res[maxID]

            # Compute error KS function of residuals
            avg = Res2max + 1/rhoKS*np.log(np.sum(np.exp(rhoKS*(Res2 - Res2max))))
           
            # Compute derivative of the residuals KS
            davg_dTheta = 2*Resmax*dRes_dTheta[:,maxID] + np.exp(rhoKS*(Res2 - Res2max)).dot(2*(Res*dRes_dTheta - Resmax*np.array([dRes_dTheta[:,maxID]]).T).T)/np.exp(rhoKS*(avg-Res2max))

        # RETURNS
        return avg, davg_dTheta

    #--------------
    
    # Define function to compute regularization factor
    def regFact(Theta_flat):

        # Get regularization contribution
        Reg = reg_factor*np.sum(Theta_flat**2)/2/len(Theta_flat)

        # RETURNS
        return Reg
    

    def regFactGrad(Theta_flat):

        # Get regularization contribution
        Reg = reg_factor*np.sum(Theta_flat**2)/2/len(Theta_flat)

        # Get regularization contribution
        dReg_dTheta = reg_factor*Theta_flat/len(Theta_flat)

        # RETURNS
        return Reg, dReg_dTheta


    #====================================================
    ## OBJECTIVE AND CONSTRAINT FUNCTIONS

    # STRATEGY 1 - avgPDE and avgBC in objective
    # STRATEGY 2 - avgPDE in objective and avgBC in constraint
    # STRATEGY 3 - avgPDE in objective and allBC in constraints

    def objFun(Theta_flat):

        if strategy == 1:
            avgPDE = avgRes(Theta_flat, Residual_PDE)
            avgBC = avgRes(Theta_flat, Residual_BC)
            
            obj = avgPDE + avgBC

        else:
            avgPDE = avgRes(Theta_flat, Residual_PDE)

            obj = avgPDE
        
        # Include regularization term
        Reg = regFact(Theta_flat)

        # Update objective with the regularization factor
        obj = obj + Reg

        # Update history if necessary
        if np.mod(num_calls[0], save_interval) == 0:

            # Save the current set of weights
            Theta_flat_hist.append(Theta_flat)
            obj_hist.append(obj)

            # Generate a plot of the current state
            if plot_function is not None:
                plot_state(NN_set, plot_function, image_dir, num_calls[0])

            # Save the history file
            if hist_file is not None:
                save_history(NN_set, Theta_flat_hist, Theta_flat_ALM_hist, obj_hist, num_calls,
                             optimizer, opt_options, reg_factor, rhoKS,
                             hist_file)
            
        num_calls[0] = num_calls[0] + 1

        # RETURNS
        return obj

    def objFunGrad(Theta_flat):

        if strategy == 1:
            avgPDE, avgPDEGrad = avgResGrad(Theta_flat, Residual_PDE)
            avgBC, avgBCGrad = avgResGrad(Theta_flat, Residual_BC)
            
            objGrad = avgPDEGrad + avgBCGrad

        else:
            avgPDE, avgPDEGrad = avgResGrad(Theta_flat, Residual_PDE)

            objGrad = avgPDEGrad
        
        # Include regularization term
        Reg, RegGrad = regFactGrad(Theta_flat)

        objGrad = objGrad + RegGrad

        # RETURNS
        return objGrad

    #--------------

    def conFun(Theta_flat):

        if strategy == 1:
            con = 0
        
        elif strategy == 2:

            avgBC = avgRes(Theta_flat, Residual_BC)
            
            con = avgBC

        elif strategy == 3:
            allBC = allRes(Theta_flat, Residual_BC)

            con = allBC # This is an array

        # RETURNS
        return con

    def conFunGrad(Theta_flat):

        if strategy == 1:

            conGrad = np.zeros_like(Theta_flat)
        
        elif strategy == 2:

            avgBC, avgBCGrad = avgResGrad(Theta_flat, Residual_BC)
            
            conGrad = avgBCGrad

        elif strategy == 3:

            allBC, allBCGrad = allResGrad(Theta_flat, Residual_BC)

            conGrad = allBCGrad.T # This is an array

        # RETURNS
        return conGrad

    #--------------
    # Define helper function to interface with the ALM optimizer
    def objConFunc(Theta_flat):

        # Compute all objectives, constraints and gradients
        obj = objFun(Theta_flat)
        objGrad = objFunGrad(Theta_flat)
        con = conFun(Theta_flat)
        conGrad = conFunGrad(Theta_flat)

        # Transform into array if necessary
        if strategy in [1,2]:
            con = np.array([con])
            conGrad = np.array([conGrad])

        # RETURNS
        return obj, objGrad, [0], np.zeros(len(Theta_flat)), con, conGrad

    if check_gradients: #Check if the user wants to verify the gradients
        
        # Define step size
        stepSize = 1e-7

        # Define random perturbation for Theta
        Thetad = np.random.rand(*Theta_flat0.shape)

        ## OBJECTIVE FUNCTION

        # Compute analytical directional derivative
        objGrad = objFunGrad(Theta_flat0)
        objGrad_an = objGrad.dot(Thetad)

        # Compute numerical gradient
        obj0 = objFun(Theta_flat0)
        obj1 = objFun(Theta_flat0 + stepSize*Thetad)
        objGrad_num = (obj1 - obj0)/stepSize

        obj_diff = np.abs(1-objGrad_num/objGrad_an)

        # Print log
        print('funcs_pde.train_pde_nn: Gradient verification requested...')
        print('Maximum absolute difference for objective function: ',obj_diff)
        if obj_diff < 1e-5:
            print('OK! Objective functions is correct!')
        else:
            print('WRONG! Verify residual functions gradients!')
            quit()

    # Start time counter for the training process
    time_start = time.time()

    # Run the optimizer
    if optimizer == 'ALM':
        
        # Run the optimization
        from .optimizers import ALM
        [Theta_flat_ALM_hist] = ALM(objConFunc, Theta_flat0, **opt_options)[0:1]

        # Get the final answer
        Theta_flatf = Theta_flat_ALM_hist[:,-1]

    else:

        # Set Bounds
        bounds = [(-10,10) for i in range(len(Theta_flat0))]

        # Run the optimizer
        if strategy == 1: # Unconstrained optimization

            cons = []


        if strategy in [2,3]: # Constrained optimization
        
            # Create constraint dictionary
            cons = {'type': 'eq',
                    'fun': conFun,
                    'jac': conFunGrad}
            
        # Run the optimization
        from scipy.optimize import minimize
        result = minimize(objFun, Theta_flat0, method=optimizer,
                          jac=objFunGrad, bounds=bounds,
                          constraints=cons,
                          options=opt_options)
        print(result)

        # Get the final answer
        Theta_flatf = result.x

    # End time counter for the training process
    time_end = time.time()

    # Compute training time
    training_time = time_end - time_start

    # Reassign weights
    reassign_theta(NN_set,Theta_flatf)

    # Save the current set of weights
    Theta_flat_hist.append(Theta_flatf)

    # Generate a plot of the current state 
    if plot_function is not None:
        plot_state(NN_set, plot_function, image_dir, num_calls[0])

    # Save final state into the pickle file
    if hist_file is not None:
        save_history(NN_set, Theta_flat_hist, Theta_flat_ALM_hist,
                     obj_hist, num_calls,
                     optimizer, opt_options, reg_factor, rhoKS,
                     hist_file)

    # RETURNS
    return obj_hist, Theta_flat_hist, training_time, Theta_flat_ALM_hist
    #NN_set.Theta_flat is updated

###########################################

###########################################

def reassign_theta(NN_set,Theta_flat):
    # This function breaks the Theta_flat array into the components of each NN

    # Prepare indices for the first assignment
    start_index = 0
    end_index = 0

    # Assign weights
    for index in range(len(NN_set)):

        # Update indices
        start_index = end_index
        end_index = start_index + NN_set[index].num_theta

        # Slice array
        NN_set[index].Theta_flat = Theta_flat[start_index:end_index]

###########################################

def use_NN_set(Inputs,NN_set):
    '''
    This function computes all outputs, sensitivities and gradients for each NN of NN_set
    and returns lists of results for each NN.
    It is useful to use with Residual_BC and Residual_PDE functions.
    IDEA: Give one input set for each NN
    '''

    # Problem size
    num_NN = len(NN_set)
    num_inputs = Inputs.shape[0]
    num_cases = Inputs.shape[1]

    # Get the total number of weights
    num_theta_tot = 0
    for index in range(num_NN):
        num_theta_tot = num_theta_tot + NN_set[index].num_theta

    # Initialize lists
    Outputs_set = [] # Initialize list to hold all outputs
    dOutputs_dTheta_set = [np.zeros((num_theta_tot, num_cases)) for dummy in range(num_NN)] # Initialize list to hold all outputs gradients. This crazy for inside is to prevent Python from creating mirrored matrices!
    dOutputs_dInputs_set = [] # Initialize list to hold all sensitivities
    dSens_dTheta_set = [np.zeros((num_theta_tot, num_inputs, num_cases)) for dummy in range(num_NN)] # Initialize list to hold all sensitivities gradients (Sens = dOutputs_dInputs)
    dOutputs_dInputsdInputs_set = [] # Initialize list to hold all sensitivities
    dHess_dTheta_set = [np.zeros((num_theta_tot, num_inputs, num_inputs, num_cases)) for dummy in range(num_NN)] # Initialize list to hold all hessian gradients (Hess = dOutputs_dInputsdInputs)

    # Initialize index for theta slicing
    start_index = 0
    end_index = 0

    # Find outputs, sensitivities and their gradients for all NNs
    for index in range(num_NN):

        # Update indices
        start_index = end_index
        end_index = start_index + NN_set[index].num_theta

        # Use backpropagation for one NN
        dOutputs_dTheta, dSens_dTheta, dHess_dTheta, Outputs, dOutputs_dInputs, dOutputs_dInputsdInputs = NN_set[index].backpropagation(Inputs)[0:6]

        # Append data to the list
        Outputs_set = Outputs_set + [Outputs]
        dOutputs_dTheta_set[index][start_index:end_index,:] = dOutputs_dTheta
        dOutputs_dInputs_set = dOutputs_dInputs_set + [dOutputs_dInputs]
        dSens_dTheta_set[index][start_index:end_index,:,:] = dSens_dTheta
        dOutputs_dInputsdInputs_set = dOutputs_dInputsdInputs_set + [dOutputs_dInputsdInputs]
        dHess_dTheta_set[index][start_index:end_index,:,:,:] = dHess_dTheta

    # RETURNS
    return Outputs_set, dOutputs_dTheta_set, dOutputs_dInputs_set, dSens_dTheta_set, dOutputs_dInputsdInputs_set, dHess_dTheta_set

###########################################

def save_history(NN_set, Theta_flat_hist, Theta_flat_ALM_hist,
                 obj_hist, num_calls,
                 optimizer, opt_options, reg_factor, rhoKS,
                 hist_file):
    '''
    This function saves a pickle file with the current optimization history.
    This file can be used to hot-start a new ANN training.
    '''

    # Create dictionary with parameters and results
    save_data = {'NN_set':NN_set,
                 'Theta_flat_hist':Theta_flat_hist,
                 'Theta_flat_ALM_hist':Theta_flat_ALM_hist,
                 'obj_hist':obj_hist,
                 'num_calls':num_calls,
                 'optimizer':optimizer,
                 'opt_options':opt_options,
                 'reg_factor':reg_factor,
                 'rhoKS':rhoKS}
        
    # Save results in a pickle file
    with open(hist_file,'wb') as fid:
        pickle.dump(save_data,fid)

###########################################

def load_history(hist_file):
    '''
    This function loads a pickle file with optimization history.
    This file can be used to hot-start a new ANN training.
    '''
        
    # Load information from a pickle file
    with open(hist_file,'rb') as fid:
        save_data = pickle.load(fid)

    # Split the dictionary
    NN_set = save_data['NN_set']
    Theta_flat_hist = save_data['Theta_flat_hist']
    Theta_flat_ALM_hist = save_data['Theta_flat_ALM_hist']
    obj_hist = save_data['obj_hist']
    num_calls = save_data['num_calls']
    optimizer = save_data['optimizer']
    opt_options = save_data['opt_options']
    reg_factor = save_data['reg_factor']
    rhoKS = save_data['rhoKS']

    return NN_set, Theta_flat_hist, obj_hist, num_calls, optimizer, opt_options, reg_factor, rhoKS, Theta_flat_ALM_hist

###########################################

def plot_state(NN_set, plot_function, image_dir, iterID):
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
        fig.savefig(os.path.join(image_dir,'image%d_%05d.png'%(ii,iterID)),dpi=300)
        plt.close(fig)

###########################################
