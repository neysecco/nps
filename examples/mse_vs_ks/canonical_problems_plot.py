from statistics import mean
from nps import NN, funcs_pde, postproc_pde
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Path to import test_cases_journal
import sys
sys.path.append('../canonical_cases/ann_solution/')
from test_cases_journal import load_case, plot_canonical

# Figure properties
exec(open("../figure_template.py").read())

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

# Define better names for each test case
text_case_names = {'lin_adv':'Linear Advection',
                   'lin_adv2':'Periodic Linear Advection',
                   'heat_eq':'Steady Heat Transfer',
                   'heat_cond':'Unsteady Heat Transfer',
                   'burgers':"Burgers' Equation"}
               

#===============================================
'''
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
'''
#===============================================

# Scatter + Histogram Plots

# definitions for the axes
left, width = 0.17, 0.6
bottom, height = 0.13, 0.6
spacing = 0.005

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# Auxiliary function to plot scatter and histograms
def scatter_hist(valMSE_MSE, training_time_MSE,
                 valMSE_KS, training_time_KS,
                 case_name,
                 ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.plot(np.log10(valMSE_MSE), training_time_MSE, 'ok', label='MSE')
    ax.plot(np.log10(valMSE_KS), training_time_KS, 'or', label='KS')

    # now determine nice limits by hand:
    MSE_min = min(np.min(valMSE_MSE), np.min(valMSE_KS))
    MSE_max = max(np.max(valMSE_MSE), np.max(valMSE_KS))
    MSE_min = 10**(np.floor(np.log10(MSE_min)))
    MSE_max = 10**(np.ceil(np.log10(MSE_max)))

    t_min = min(np.min(training_time_MSE), np.min(training_time_KS))
    t_max = max(np.max(training_time_MSE), np.max(training_time_KS))

    ax_histx.hist(np.log10(valMSE_MSE), 10, range=[np.log10(MSE_min), np.log10(MSE_max)], facecolor='k', alpha=0.75)
    ax_histx.hist(np.log10(valMSE_KS), 10, range=[np.log10(MSE_min), np.log10(MSE_max)], facecolor='r', alpha=0.75)
    ax_histy.hist(training_time_MSE, 10, range=[t_min, t_max], facecolor='k', alpha=0.75, orientation='horizontal')
    ax_histy.hist(training_time_KS, 10, range=[t_min, t_max], facecolor='r', alpha=0.75, orientation='horizontal')

    # Remove extra lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(r'$MSE_\textrm{val}$')
    ax.set_ylabel(r'$T \; [s]$')

    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)
    ax_histx.tick_params(axis='x', which=u'both', length=0, labelbottom=False)
    ax_histx.tick_params(axis='y', which=u'both', length=0, labelleft=False)

    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.spines['bottom'].set_visible(False)
    ax_histy.tick_params(axis='x', which=u'both', length=0, labelbottom=False)
    ax_histy.tick_params(axis='y', which=u'both', length=0, labelleft=False)

    ax_histx.set_title(text_case_names[case_name])

    # Set powers of 10 in the MSE axis
    labels=ax.get_xticks().tolist()
    labels = [r'$10^{%d}$'%item for item in labels]
    ax.set_xticklabels(labels)

    # Add legend only to the linear advection plot
    if case_name == 'lin_adv':
        ax.legend(loc='best')

# Loop for every canonical case
for ii in range(num_cases):

    fig = plt.figure(figsize=(6.5, 7))

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    scatter_hist(valMSE_MSE[ii], training_time_MSE[ii],
                 valMSE_KS[ii], training_time_KS[ii],
                 canonical_prob[ii],
                 ax, ax_histx, ax_histy)

    fig.savefig('%s_scatter.pdf'%canonical_prob[ii])

#===============================================

plt.tight_layout()
plt.show()