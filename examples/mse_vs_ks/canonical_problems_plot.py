from statistics import mean
from nps import NN, funcs_pde, postproc_pde
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Path to import test_cases_journal
import sys
sys.path.append('../canonical_cases/ann_solution/')
from test_cases_journal import load_case, plot_canonical

with open('./sweep_results.pickle','rb') as fid:
    results = pickle.load(fid)

valMSE_MSE = results[0]
training_time_MSE = results[1]
valMSE_KS = results[2]
training_time_KS = results[3]
canonical_prob = results[4]

print(valMSE_MSE)
print(valMSE_KS)

# Get sizes
num_cases, num_theta = valMSE_MSE.shape

#===============================================

# Initialize figure

fig = plt.figure()

for ii in range(num_cases):

    # Get average values corresponding to the current case
    valMSE_MSE_avg = np.prod(valMSE_MSE[ii])**(1/num_theta)
    valMSE_KS_avg = np.prod(valMSE_KS[ii])**(1/num_theta)

    # MSEval lines

    ax = plt.subplot(3, num_cases, ii+1)

    # Get range of values
    MSE_min = min(np.min(valMSE_MSE[ii]), np.min(valMSE_KS[ii]))
    MSE_max = max(np.max(valMSE_MSE[ii]), np.max(valMSE_KS[ii]))
    MSE_min = 10**(np.floor(np.log10(MSE_min)))
    MSE_max = 10**(np.ceil(np.log10(MSE_max)))

    # Reference lines
    plt.semilogx([MSE_min, MSE_max], [0, 0], 'gray')#,linewidth=0.2)
    plt.semilogx([MSE_min, MSE_max], [1, 1], 'gray')#,linewidth=0.2)

    # Scatter of all values measured
    plt.semilogx(valMSE_MSE[ii], [1]*num_theta, 'ok')
    plt.semilogx(valMSE_KS[ii], [0]*num_theta, 'ok')

    # Average values
    plt.semilogx([valMSE_MSE_avg], [1], '^r')
    plt.semilogx([valMSE_KS_avg], [0], '^r')

    # Set axes names
    if ii==0:
        plt.yticks([0, 1], ['KS', 'MSE'], fontsize=12)
    else:
        plt.yticks([], [], fontsize=12)
    plt.title(canonical_prob[ii],fontsize=14)

    # Adjust splines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.minorticks_off()
    ax.tick_params(axis='y', which=u'both',length=0)

    #==========================================
    # Time lines

    ax = plt.subplot(3, num_cases, 2*num_cases+ii+1)

    # Get range of values
    t_min = min(np.min(training_time_MSE[ii]), np.min(training_time_KS[ii]))
    t_max = max(np.max(training_time_MSE[ii]), np.max(training_time_KS[ii]))

    # Reference lines
    plt.plot([t_min, t_max], [0, 0], 'gray')#,linewidth=0.2)
    plt.plot([t_min, t_max], [1, 1], 'gray')#,linewidth=0.2)

    # Scatter of all values measured
    plt.plot(training_time_MSE[ii], [1]*num_theta, 'ok')
    plt.plot(training_time_KS[ii], [0]*num_theta, 'ok')

    # Average values
    plt.plot(np.mean(training_time_MSE[ii]), [1], '^r')
    plt.plot(np.mean(training_time_KS[ii]), [0], '^r')

    # Set axes names
    if ii==0:
        plt.yticks([0, 1], ['KS', 'MSE'], fontsize=12)
    else:
        plt.yticks([], [], fontsize=12)

    # Adjust splines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.minorticks_off()
    ax.tick_params(axis='y', which=u'both',length=0)

#===============================================

# Initialize figure

fig = plt.figure()

for ii in range(num_cases):

    # Get average values corresponding to the current case
    valMSE_MSE_avg = np.prod(valMSE_MSE[ii])**(1/num_theta)
    valMSE_KS_avg = np.prod(valMSE_KS[ii])**(1/num_theta)

    # MSEval lines

    ax = plt.subplot(1, num_cases, ii+1)

    # Scatter of all values measured
    plt.semilogx(valMSE_MSE[ii], training_time_MSE[ii], 'ok')
    plt.semilogx(valMSE_KS[ii], training_time_KS[ii], 'or')

    # Set axes names
    if ii==0:
        plt.ylabel('T [s]', fontsize=12)
    plt.xlabel('MSEval', fontsize=12)
    plt.title(canonical_prob[ii],fontsize=14)

    # Adjust splines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #plt.minorticks_off()
    #ax.tick_params(axis='y', which=u'both',length=0)

#===============================================

# Initialize figure

fig = plt.figure()

for ii in range(num_cases):

    # Remove one outlier from the heat_eq case
    if ii == 2:
        for jj in range(len(valMSE_KS[ii])):
            if valMSE_KS[ii][jj] > 1e-2:
                valMSE_KS[ii][jj] = None

    # Get average values corresponding to the current case
    valMSE_MSE_avg = np.prod(valMSE_MSE[ii])**(1/num_theta)
    valMSE_KS_avg = np.prod(valMSE_KS[ii])**(1/num_theta)

    # MSEval lines

    ax = plt.subplot(2, num_cases, ii+1)

    # Get range of values
    MSE_min = min(np.min(valMSE_MSE[ii]), np.min(valMSE_KS[ii]))
    MSE_max = max(np.max(valMSE_MSE[ii]), np.max(valMSE_KS[ii]))
    MSE_min = 10**(np.floor(np.log10(MSE_min)))
    MSE_max = 10**(np.ceil(np.log10(MSE_max)))

    # Scatter of all values measured
    plt.hist(np.log10(valMSE_MSE[ii]), 10, range=[np.log10(MSE_min), np.log10(MSE_max)], facecolor='k', alpha=0.75)
    plt.hist(np.log10(valMSE_KS[ii]), 10, range=[np.log10(MSE_min), np.log10(MSE_max)], facecolor='r', alpha=0.75)

    # Set axes names
    if ii==0:
        plt.ylabel('Frequency', fontsize=12)
    plt.xlabel('MSEval', fontsize=12)
    plt.title(canonical_prob[ii],fontsize=14)

    # Adjust splines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #==========================================
    # Time lines

    ax = plt.subplot(2, num_cases, num_cases+ii+1)

    # Get range of values
    t_min = min(np.min(training_time_MSE[ii]), np.min(training_time_KS[ii]))
    t_max = max(np.max(training_time_MSE[ii]), np.max(training_time_KS[ii]))

    # Scatter of all values measured
    plt.hist(training_time_MSE[ii], 10, range=[t_min, t_max], facecolor='k', alpha=0.75)
    plt.hist(training_time_KS[ii], 10, range=[t_min, t_max], facecolor='r', alpha=0.75)

    # Set axes names
    if ii==0:
        plt.ylabel('Frequency', fontsize=12)
    plt.xlabel('T [s]', fontsize=12)
    plt.title(canonical_prob[ii],fontsize=14)

    # Adjust splines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

#===============================================

plt.tight_layout()
plt.show()