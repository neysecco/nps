'''
This script performs the scalability analysis of the EFP and
EBP methods presented in the article.

This script can either load timings already stored in a
pickle file, or measure new execution times to generate
a new pickle file.
1) If you set 'run_cases = False', this script will generate
plots with the timings present in the pickle file.
2) If you set 'run_cases = True', then the pickle file will
be replaced with new timings. The execution might take a while.

After you select the run_cases flag, you can execute this script with:
$ python3 main.py
'''

# Select if we will run the cases once again (True) or
# reload previous results (False)
run_cases = False

# IMPORTS
from nps import NN
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

#========================================

# AUXILIARY FUNCTIONS

def time_ann(num_inputs, num_neurons, num_layers, num_cases,
             num_runs=10):
    '''
    This function measures average runtimes for EFP and EBP
    for a given network

    INPUTS
    num_inputs: int -> number of neurons on the input layer
    num_neurons: int -> number of hidden neurons on each hidden layer
    num_layer: int -> number of hidden layers
    num_cases: int -> number of cases (sampling points)
    num_runs: int -> number of executions to take averaged time
    '''

    # Define upper and lower bounds of the domain
    lb = -1.0
    ub = +1.0

    # Generate Neural Network
    NN_test = NN(num_inputs=num_inputs, num_outputs=1,
                 num_neurons_list=[num_neurons]*num_layers,
                 lower_bounds=[lb]*num_inputs, upper_bounds=[ub]*num_inputs)

    # Take samples within the domain
    Inputs = lb + (ub-lb)*np.random.rand(num_inputs, num_cases)

    # FEEDFORWARD

    # Start counter
    counter = 0
    start_time = time.perf_counter()

    # Loop
    while counter < num_runs:
        NN_test.feedforward(Inputs)
        counter = counter + 1

    # End counter
    end_time = time.perf_counter()

    # Compute averaged time
    efp_time = (end_time - start_time)/num_runs

    # BACKPROPAGATION

    # Start counter
    counter = 0
    start_time = time.perf_counter()

    # Loop
    while counter < num_runs:
        NN_test.backpropagation(Inputs)
        counter = counter + 1

    # End counter
    end_time = time.perf_counter()

    # Compute averaged time
    ebp_time = (end_time - start_time)/num_runs

    # Return measured times
    return efp_time, ebp_time

#============================================

def execute_tests(ref_inputs=2, ref_neurons=5, ref_layers=2, ref_cases=5,
                  scale_list=[1,3,5,10], num_runs=10,
                  param_list = ['num_neurons', 'num_layers', 'num_cases']):
    '''
    INPUTS
    num_inputs: int -> number of neurons on the input layer of the reference ANN
    num_neurons: int -> number of hidden neurons on each hidden layer of the reference ANN
    num_layer: int -> number of hidden layers of the reference ANN
    num_cases: int -> number of cases (sampling points) of the reference ANN
    scale: list of int -> scale factors to be applied at each parameter
    param_list: list of strings -> which parameters from time_ann inputs that should be scaled
    '''

    # Dictionary of reference values of inputs for the time_ann function
    time_ann_inputs_ref = {'num_inputs':ref_inputs,
                           'num_neurons':ref_neurons,
                           'num_layers':ref_layers,
                           'num_cases':ref_cases,
                           'num_runs':num_runs}

    # Select which parameters should be varied
    

    # Time the reference ANN (have to do twice due to setup time)
    ref_efp_time, ref_ebp_time = time_ann(**time_ann_inputs_ref)
    ref_efp_time, ref_ebp_time = time_ann(**time_ann_inputs_ref)

    # Initialize results dictionary
    efp_times = {s:np.zeros(len(scale_list)) for s in param_list}
    ebp_times = {s:np.zeros(len(scale_list)) for s in param_list}
    ebp_efp_ratio = {s:np.zeros(len(scale_list)) for s in param_list}

    # Scale each input
    for input_name in param_list:

        # Make a copy of the reference dictionary
        time_ann_inputs = time_ann_inputs_ref.copy()

        # Apply each scaling factor
        for ii,scale in enumerate(scale_list):

            # Update dictionary with scaled parameter
            time_ann_inputs[input_name] = scale*time_ann_inputs_ref[input_name]

            # Time the reference ANN
            efp_time, ebp_time = time_ann(**time_ann_inputs)

            # Store normalized results
            efp_times[input_name][ii] = efp_time/ref_efp_time
            ebp_times[input_name][ii] = ebp_time/ref_ebp_time#/efp_times[input_name][ii]
            ebp_efp_ratio[input_name][ii] = ebp_time/efp_time

    return efp_times, ebp_times, ebp_efp_ratio

#============================================

# EXECUTION

# Define a name for the pickle file that wil store results
results_file = 'timings.pickle'

# Check if we will replace the timings from the pickle file
if run_cases:

    # Set seed to avoid randomness
    np.random.seed(123)

    # Select standard number of ANN inputs
    ref_inputs = 2
    ref_neurons = 5

    # Define list of scale factors
    scale_list = [1, 2, 3, 5, 10, 20, 30, 50, 100]
    param_list = ['num_neurons', 'num_cases']

    # Execute all test cases
    efp_times, ebp_times, ebp_efp_ratio = execute_tests(ref_inputs=ref_inputs,
                                                        ref_neurons=ref_neurons,
                                                        scale_list=scale_list,
                                                        num_runs=500,
                                                        param_list=param_list)

    # Save results in a pickle file
    with open(results_file,'wb') as fid:
        pickle.dump([ref_inputs,
                     ref_neurons,
                     scale_list,
                     efp_times,
                     ebp_times,
                     ebp_efp_ratio],fid)

else: # Use results stored in the pickle file

    # Load results from the pickle file
    with open(results_file,'rb') as fid:
        [ref_inputs,
         ref_neurons,
         scale_list,
         efp_times,
         ebp_times,
         ebp_efp_ratio] = pickle.load(fid)

# Plot results - EFP
exec(open("../figure_template.py").read())
fig = plt.figure()
for key,times in efp_times.items():
    
    plt.loglog(scale_list, times, '-o', label=r'$\mathrm{%s}$'%key.replace('_',' \; '))
    
    # Special treatment to convert number of neurons into number of weights
    if key == 'num_neurons':

        # Compute number of weights for the reference ANN
        ref_theta = ref_neurons*(ref_inputs+1)+(ref_neurons+1)**2

        # Convert the neurons scale factor into weights scale factor
        num_theta = [(n*ref_neurons*(ref_inputs+1)+(n*ref_neurons+1)**2)/ref_theta for n in scale_list]

        # Plot results
        plt.loglog(num_theta, times, '-o', label=r'$\mathrm{%s}$'%'num \; weights')

# Add text labels
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.text(0.9, 678, r'$\bm{\mathrm{\# \; of \; neurons \atop per \; hidden \; layer}}$', color=colors[0])
plt.text(500, 150, r'$\bm{\mathrm{\# \; of \; weights \atop}}$', color=colors[1])
plt.text(55, 8.2, r'$\bm{\mathrm{\# \; of \; sampling \; points \atop}}$', color=colors[2])

# Plot linear trendline
plt.loglog([1e0, 1e3],[1e0, 1e3], '--k')
plt.text(100, 120, r'$\bm{\mathrm{linear \; trend \atop}}$', color='k', rotation=36)

#plt.legend(loc='best')
plt.xlabel(r'$\mathrm{scale \; factor}$')
plt.ylabel(r'$\mathrm{normalized \; time}$')
plt.tight_layout()

fig.savefig('efp_time.pdf')

#=================================================

# Plot results - EBP
exec(open("../figure_template.py").read())
fig = plt.figure()
for key,times in ebp_times.items():
    
    plt.loglog(scale_list, times, '-o', label=r'$\mathrm{%s}$'%key.replace('_',' \; '))
    
    # Special treatment to convert number of neurons into number of weights
    if key == 'num_neurons':

        # Compute number of weights for the reference ANN
        ref_theta = ref_neurons*(ref_inputs+1)+(ref_neurons+1)**2

        # Convert the neurons scale factor into weights scale factor
        num_theta = [(n*ref_neurons*(ref_inputs+1)+(n*ref_neurons+1)**2)/ref_theta for n in scale_list]

        # Plot results
        plt.loglog(num_theta, times, '-o', label=r'$\mathrm{%s}$'%'num \; weights')

# Add text labels
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.text(0.9, 678, r'$\bm{\mathrm{\# \; of \; neurons \atop per \; hidden \; layer}}$', color=colors[0])
plt.text(500, 150, r'$\bm{\mathrm{\# \; of \; weights \atop}}$', color=colors[1])
plt.text(55, 8.2, r'$\bm{\mathrm{\# \; of \; sampling \; points \atop}}$', color=colors[2])

# Plot linear trendline
plt.loglog([1e0, 1e3],[1e0, 1e3], '--k')
plt.text(100, 120, r'$\bm{\mathrm{linear \; trend \atop}}$', color='k', rotation=36)

#plt.legend(loc='best')
plt.xlabel(r'$\mathrm{scale \; factor}$')
plt.ylabel(r'$\mathrm{normalized \; time}$')
plt.tight_layout()

fig.savefig('ebp_time.pdf')

#=================================================

# Plot results - EBP/EFP
exec(open("../figure_template.py").read())
fig = plt.figure()
for key,times in ebp_efp_ratio.items():
    
    plt.loglog(scale_list, times, '-o', label=r'$\mathrm{%s}$'%key.replace('_',' \; '))
    
    # Special treatment to convert number of neurons into number of weights
    if key == 'num_neurons':

        # Compute number of weights for the reference ANN
        ref_theta = ref_neurons*(ref_inputs+1)+(ref_neurons+1)**2

        # Convert the neurons scale factor into weights scale factor
        num_theta = [(n*ref_neurons*(ref_inputs+1)+(n*ref_neurons+1)**2)/ref_theta for n in scale_list]

        # Plot results
        plt.loglog(num_theta, times, '-o', label=r'$\mathrm{%s}$'%'num \; weights')

# Add text labels
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.text(30, 45, r'$\bm{\mathrm{\# \; of \; neurons \atop per \; hidden \; layer}}$', color=colors[0])
plt.text(500, 15, r'$\bm{\mathrm{\# \; of \; weights \atop}}$', color=colors[1])
plt.text(55, 2.7, r'$\bm{\mathrm{\# \; of \; sampling \; points \atop}}$', color=colors[2])

# Plot linear trendline
print(ebp_efp_ratio.keys())
key0 = list(ebp_efp_ratio.keys())[0]
print(key0)
time0 = ebp_efp_ratio[key0][0]
plt.loglog([1e0, 1e2],[time0, time0*1e2], '--k')
plt.text(4.0, 10, r'$\bm{\mathrm{linear \; trend \atop}}$', color='k', rotation=50)

#plt.legend(loc='best')
plt.xlabel(r'$\mathrm{scale \; factor}$')
plt.ylabel(r'$\mathrm{EBP \; time \; / \; EFP \; time}$')
plt.tight_layout()
plt.show()

fig.savefig('ebp_efp_time.pdf')