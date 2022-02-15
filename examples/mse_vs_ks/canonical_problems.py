from nps import NN, funcs_pde, postproc_pde
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to import test_cases_journal
import sys
sys.path.append('../canonical_cases/ann_solution/')
from test_cases_journal import load_case, plot_canonical

canonical_prob   = ['lin_adv', 'lin_adv2','heat_eq','heat_cond','burgers']
minor_iter  = [2500,2500,2500,2500,2500]
major_iter  = [10,10,10,10,10,10,10]
forced_iter = 0
boundary_points = [31,31,31,31,31]
domain_points   = [21,21,31,21,31]
num_neurons     = [[5,5],[5,5],[5,5,5],[5,5],[5,5,5]]
axis_label      = [['x','t'],['x','t'],['x','y'],['x','t'],['x','t']]

def eval_case(theta_seed, canonical_prob,
              minor_iter, major_iter,
              boundary_points, domain_points,
              num_neurons, axis_label,
              metric='MSE', interval=100000000):
    
    # Problem options
    dist_type = 'unstructured' # 'unstructured' | 'structured' 
    optimizer = 'ALM'
    strategy  = 3

    # Get problem definition
    [NN_set,
    Residual_BC,
    Residual_PDE,
    plot_function,
    val_function,
    MSEhist_function,
    reg_factor,
    rhoKS,
    opt_options,
    hist_file,
    image_dir,
    Inputs_do,
    Inputs_bc,
    Targets_bc] = load_case(canonical_prob, dist_type, optimizer,boundary_points,domain_points,
                            clean_image_dir=True,num_neurons=num_neurons,Theta_seed=theta_seed,axis_label=axis_label)
    
    # Modify the initial seed.
    training_time = 0.0
    rhoKS         = 0.1
    reg_factor    = 0.0
    
    opt_options['delta_x_tol'] = 1e-10
    opt_options['grad_tol'] = 1e-10
    opt_options['minor_iterations'] = minor_iter
    opt_options['major_iterations'] = major_iter
    opt_options['grad_opt'] = 'BFGS'

    ###########################################
    # Optimization with BFGS
        
    # Solve PDE
    obj_hist, Theta_flat_hist, training_time, Theta_hist_ALM = funcs_pde.train_pde_nn(NN_set, Residual_BC, 
                                                                      Residual_PDE,
                                                                      optimizer = optimizer,
                                                                      metric = metric,
                                                                      opt_options = opt_options,
                                                                      reg_factor = reg_factor,
                                                                      rhoKS = rhoKS,
                                                                      check_gradients = False,
                                                                      strategy = strategy,
                                                                      hist_file = hist_file,
                                                                      save_interval = interval,
                                                                      load_hist = False,
                                                                      plot_function = plot_function,
                                                                      image_dir = image_dir)
                                                                         
    valMSE = val_function(NN_set)
    print('\n\n#Plotting results for the '+canonical_prob+' problem\n')
    plot_canonical(NN_set, plot_function, './'+ canonical_prob + '/'+ canonical_prob)
    MSEhist_function(NN_set,Theta_flat_hist,Theta_hist_ALM,1)
    
    return valMSE, training_time
        

#############################################

# Number of seeds
num_theta_seed = 20

theta_seed = list(range(num_theta_seed))

valMSE_MSE = np.zeros((len(canonical_prob), num_theta_seed))
training_time_MSE = np.zeros((len(canonical_prob), num_theta_seed))
valMSE_KS = np.zeros((len(canonical_prob), num_theta_seed))
training_time_KS = np.zeros((len(canonical_prob), num_theta_seed))

for ii in range(len(canonical_prob)):
    for jj in range(num_theta_seed):

        [valMSE_MSE[ii,jj], training_time_MSE[ii,jj]] = eval_case(theta_seed[jj],
                                                                  canonical_prob[ii], 
                                                                  minor_iter[ii],
                                                                  major_iter[ii],
                                                                  boundary_points[ii],
                                                                  domain_points[ii],
                                                                  num_neurons[ii],
                                                                  axis_label[ii],
                                                                  metric='MSE')

        # Rename directory to store its current results
        os.rename(canonical_prob[ii], canonical_prob[ii]+'_MSE_%02d'%jj)

        [valMSE_KS[ii,jj], training_time_KS[ii,jj]] = eval_case(theta_seed[jj],
                                                                canonical_prob[ii], 
                                                                minor_iter[ii],
                                                                major_iter[ii],
                                                                boundary_points[ii],
                                                                domain_points[ii],
                                                                num_neurons[ii],
                                                                axis_label[ii],
                                                                metric='KS')

        # Rename directory to store its current results
        os.rename(canonical_prob[ii], canonical_prob[ii]+'_KS_%02d'%jj)
        
        # Save intermediary results                                                          
        with open('./sweep_results.pickle','wb') as fid:
            results = [valMSE_MSE, training_time_MSE, valMSE_KS, training_time_KS,
                       canonical_prob, minor_iter, major_iter,
                       forced_iter, theta_seed, boundary_points,
                       domain_points, num_neurons, axis_label]
            pickle.dump(results, fid)
