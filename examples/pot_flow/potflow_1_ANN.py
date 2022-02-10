from nps import NN, funcs_pde, postproc_pde
import sys
sys.path.append('../canonical_cases/ann_solution/')
from test_cases_journal import load_case
import pickle
import matplotlib.pyplot as plt

# Problem options
case_name = 'potflow_singlenet'
dist_type = 'unstructured'
optimizer = 'ALM'
metric = 'MSE'
strategy = 3

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
 Targets_bc] = load_case(case_name, dist_type, optimizer,0,0,
                         clean_image_dir=True)

###########################################

# Solve PDE
obj_hist, Theta_flat_hist, training_time, Theta_flat_ALM_hist = funcs_pde.train_pde_nn(NN_set, Residual_BC, Residual_PDE,
                                                                  optimizer = optimizer,
                                                                  metric = metric,
                                                                  opt_options = opt_options,
                                                                  reg_factor = reg_factor,
                                                                  rhoKS = rhoKS,
                                                                  check_gradients = True,
                                                                  strategy = strategy,
                                                                  hist_file = hist_file,
                                                                  load_hist = False,
                                                                  plot_function = plot_function,
                                                                  image_dir = image_dir,
                                                                  save_interval = 1e10)

###########################################

valMSE = val_function(NN_set)
print('valMSE:',valMSE)
print('training_time:',training_time)

# Generate plots with the plot_function
fig_list = plot_function(NN_set)
MSEhist_function(NN_set,Theta_flat_hist,Theta_flat_ALM_hist,1)

plt.show()

