from nps import NN, funcs_pde, postproc_pde
import sys
sys.path.append('../canonical_cases/ann_solution/')
from test_cases_journal import load_case
import pickle
import matplotlib.pyplot as plt

# Problem options
case_name = 'potflow_doublenet'
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

with open('./potflow_doublenet/results.pickle','rb') as fid:
    results = pickle.load(fid)
###########################################
NN_set = results['NN_set']
Theta_flat_hist      = results['Theta_flat_hist']
Theta_flat_ALM_hist  = results['Theta_flat_ALM_hist']
valMSE = val_function(NN_set)
print('valMSE:',valMSE)

# Generate plots with the plot_function
fig_list = plot_function(NN_set)
MSEhist_function(NN_set,Theta_flat_hist,Theta_flat_ALM_hist,1)

plt.show()


