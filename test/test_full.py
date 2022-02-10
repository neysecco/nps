from neural_nets import NN, funcs_regression, funcs_testing
import numpy as np

############################################
#MAIN PARAMETERS
num_samples = 5 #Samples in each axis
num_neurons1 = 5 #Neurons in the first hidden layer
num_neurons2 = 5 #Neurons in the second hidden layer
num_neurons3 = 0

opt_options = {'display':False,
               'iterations':1000}
############################################

############################################
#PRINT HEADER
print('############################################')
print('    REGRESSION TEST (regression_test.py)    ')
print('')
############################################

############################################
#Define test-case for regression
x1_train = np.linspace(-2,2,num_samples)
x2_train = np.linspace(-2,2,num_samples)

x1_test = np.linspace(-2,2,41)
x2_test = np.linspace(-2,2,41)

X1_train, X2_train = np.meshgrid(x1_train,x2_train)
X1_test, X2_test = np.meshgrid(x1_test,x2_test)

def trial_function(X1,X2):
        Y = (X1**2 + X2**2 + 25.0*((np.sin(X1))**2 + (np.sin(X2))**2))/40
        return Y

def trial_function_grad(X1,X2):
        dYdX1 = X1/20 + 5/8*np.sin(2*X1)
        dYdX2 = X2/20 + 5/8*np.sin(2*X2)
        return dYdX1, dYdX2

Y_train = trial_function(X1_train,X2_train)
dYdX1_train,dYdX2_train = trial_function_grad(X1_train,X2_train)
Y_test = trial_function(X1_test,X2_test)
dYdX1_test,dYdX2_test = trial_function_grad(X1_test,X2_test)

x_train = np.hstack([X1_train.reshape((np.prod(X1_train.shape),1)),
                     X2_train.reshape((np.prod(X2_train.shape),1))])
y_train = Y_train.reshape((np.prod(Y_train.shape),1))
dydx1_train = dYdX1_train.reshape((-1,1))
dydx2_train = dYdX2_train.reshape((-1,1))
dydx_train = np.hstack([dydx1_train,dydx2_train])

x_test = np.hstack([X1_test.reshape((np.prod(X1_test.shape),1)),
                    X2_test.reshape((np.prod(X2_test.shape),1))])

y_test = Y_test.reshape((np.prod(Y_test.shape),1))
dydx1_test = dYdX1_test.reshape((-1,1))
dydx2_test = dYdX2_test.reshape((-1,1))
dydx_test = np.hstack([dydx1_test,dydx2_test])

x_train = x_train.T
y_train = y_train.T
dydx_train = dydx_train.T
x_test = x_test.T
y_test = y_test.T
dydx_test = dydx_test.T
############################################

############################################
#DEFINE THE NETWORK

#Size of the problem
num_cases = x_train.shape[1]
num_inputs = x_train.shape[0]
num_outputs = y_train.shape[0]

if num_neurons3 != 0:
        num_theta = num_neurons1*(num_inputs+1) + num_neurons2*(num_neurons1+1) + num_neurons3*(num_neurons2+1) + num_outputs*(num_neurons3+1)
else:
        num_theta = num_neurons1*(num_inputs+1) + num_neurons2*(num_neurons1+1) +  num_outputs*(num_neurons2+1)

#Generating initial set of weights so that the test is not random
Theta_flat = np.linspace(-1,1,num_theta)

#Generate Neural Network
if num_neurons3 != 0:
        NN_test = NN(num_inputs=2, num_outputs=1,
                     num_neurons_list=[num_neurons1, num_neurons2, num_neurons3],
                     Theta_flat=Theta_flat, lower_bounds=[-2,-2], upper_bounds=[2,2])
else:
        NN_test = NN(num_inputs=2, num_outputs=1,
                     num_neurons_list=[num_neurons1, num_neurons2],
                     Theta_flat=Theta_flat, lower_bounds=[-2,-2], upper_bounds=[2,2])

############################################

############################################
#CHECK FEEDFORWARD, SENSITIVITIES AND HESSIAN

#Get outputs and sensitivites from feedforward function
Outputs_ff, Sens_ff, Hess_ff = NN_test.feedforward(x_train)

#Get outputs and sensitivities from the backpropagation function
Outputs_bp, Sens_bp, Hess_bp = NN_test.backpropagation(x_train)[3:]

#Get sensitivities using finite differences
Sens_fd = funcs_testing.test_num_sensitivities(x_train,NN_test)

#Get hessian using finite differences
Hess_fd = funcs_testing.test_num_hessian(x_train,NN_test)

#PRINTING
#Checking feedforward
ff_bp_diff = np.max(np.abs(Outputs_ff-Outputs_bp))

print('Checking Feedforward: feedforward vs backpropagation')
if ff_bp_diff < 1e-10:
        print('OK! ',ff_bp_diff)
else:
        print('WRONG... ',ff_bp_diff)

#Checking sensitivities
fd_bp_diff = np.sqrt(np.max(np.sum((Sens_fd-Sens_bp)**2,axis=0)))
fd_ff_diff = np.sqrt(np.max(np.sum((Sens_fd-Sens_ff)**2,axis=0)))
bp_ff_diff = np.sqrt(np.max(np.sum((Sens_bp-Sens_ff)**2,axis=0)))

print('Checking Sensitivities: test_num_sensitivities vs backpropagation')
if fd_bp_diff < 1e-8:
        print('OK! ',fd_bp_diff)
else:
        print('WRONG... ',fd_bp_diff)

print('Checking Sensitivities: test_num_sensitivities vs feedforward')
if fd_ff_diff < 1e-8:
        print('OK! ',fd_ff_diff)
else:
        print('WRONG... ',fd_ff_diff)

print('Checking Sensitivities: backpropagation vs feedforward')
if bp_ff_diff < 1e-10:
        print('OK! ',bp_ff_diff)
else:
        print('WRONG... ',bp_ff_diff)

#Checking hessian
bp_fd_diff = np.max(np.abs(Hess_bp-Hess_fd))
ff_fd_diff = np.max(np.abs(Hess_ff-Hess_fd))
bp_ff_diff = np.max(np.abs(Hess_bp-Hess_ff))

print('Checking Hessian: test_num_hessian vs backpropagation')
if bp_fd_diff < 1e-7:
        print('OK! ',bp_fd_diff)
else:
        print('WRONG... ',bp_fd_diff)

print('Checking Hessian: test_num_hessian vs feedforward')
if ff_fd_diff < 1e-7:
        print('OK! ',ff_fd_diff)
else:
        print('WRONG... ',ff_fd_diff)

print('Checking Hessian: feedforward vs backpropagation')
if bp_ff_diff < 1e-10:
        print('OK! ',bp_ff_diff)
else:
        print('WRONG... ',bp_ff_diff)
############################################

############################################
#CHECK BACKPROPAGATION

#Get outputs, sensitivities, and hessian gradients w.r.t. weights from the backpropagation function
dOutputs_dTheta_bp, dSens_dTheta_bp, dHess_dTheta_bp  = NN_test.backpropagation(x_train)[:3]

#Get outputs gradients w.r.t. weights using finite differences
dOutputs_dTheta_fd = funcs_testing.test_num_grad(x_train,NN_test)

#Get sensitivities gradients w.r.t. weights using finite differences
dSens_dTheta_fd = funcs_testing.test_num_sens_grad(x_train,NN_test)

#Get hessian gradients w.r.t. weights using finite differences
dHess_dTheta_fd = funcs_testing.test_num_hess_grad(x_train,NN_test)

#PRINTING
#Checking outputs gradients
fd_bp_diff = np.max(np.abs(dOutputs_dTheta_fd-dOutputs_dTheta_bp))

print('Checking dOutputs_dTheta: test_num_sens_grad vs backpropagation')
if fd_bp_diff < 1e-7:
        print('OK! ',fd_bp_diff)
else:
        print('WRONG... ',fd_bp_diff)

#Checking sensitivities gradients
fd_bp_diff = np.max(np.abs(dSens_dTheta_fd-dSens_dTheta_bp))

print('Checking dSens_dTheta: test_num_sens_grad vs backpropagation')
if fd_bp_diff < 1e-7:
        print('OK! ',fd_bp_diff)
else:
        print('WRONG... ',fd_bp_diff)

#Checking hessian gradients
fd_bp_diff = np.max(np.abs(dHess_dTheta_fd-dHess_dTheta_bp))

print('Checking dHess_dTheta: test_num_sens_grad vs backpropagation')
if fd_bp_diff < 1e-7:
        print('OK! ',fd_bp_diff)
else:
        print('WRONG... ',fd_bp_diff)
############################################

############################################
#CHECK MSE AND BACKPROPAGATION GRADIENTS

#Finite differences
MSE_fd, Grad_fd, MSE_sens_fd, Grad_sens_fd = funcs_testing.test_num_backprop(x_train,y_train,dydx_train,NN_test)

#Backpropagation
MSE_bp, Grad_bp = funcs_regression.MSE_values(x_train,y_train,NN_test)
MSE_sens_bp, Grad_sens_bp = funcs_regression.MSE_sensitivities(x_train,y_train,dydx_train,NN_test)

#PRINT RESULTS

#Checking MSE without sensitivities
fd_bp_diff = np.abs(MSE_fd - MSE_bp)

print('Checking MSE w/o sensitivities: test_num_backprop vs MSE_values')
if fd_bp_diff < 1e-10:
        print('OK! ',fd_bp_diff)
else:
        print('WRONG... ',fd_bp_diff)

#Checking MSE with sensitivities
fd_bp_diff = np.abs(MSE_sens_fd - MSE_sens_bp)

print('Checking MSE with sensitivities: test_num_backprop vs MSE_sensitivities')
if fd_bp_diff < 1e-8:
        print('OK! ',fd_bp_diff)
else:
        print('WRONG... ',fd_bp_diff)

#Checking MSE gradients without sensitivities
fd_bp_diff = np.max(np.abs(Grad_fd-Grad_bp))

print('Checking MSE grads w/o sensitivities: test_num_backprop vs MSE_values')
if fd_bp_diff < 1e-7:
        print('OK! ',fd_bp_diff)
else:
        print('WRONG... ',fd_bp_diff)

#Checking MSE gradients with sensitivities
fd_bp_diff = np.max(np.abs(Grad_sens_fd-Grad_sens_bp))

print('Checking MSE grads with sensitivities: test_num_backprop vs MSE_sensitivities')
if fd_bp_diff < 1e-7:
        print('OK! ',fd_bp_diff)
else:
        print('WRONG... ',fd_bp_diff)
############################################

############################################
#CHECK TRAINING

#Train the ANN without sensitivities
MSE_hist = funcs_regression.train_nn(x_train, y_train, NN_test, opt_options)[0]

#Use feedforward
y_out_train_nosens = NN_test.feedforward(x_train)[0]
y_out_test_nosens = NN_test.feedforward(x_test)[0]

#Compare final MSE
MSE_diff = np.abs(MSE_hist[-1] - 7.028758414720471e-07) #5-5 neurons, 5-5 samples, no sens

print('Check Training without sensitivities:')
if MSE_diff < 1e-13:
        print('OK! ',MSE_diff)
else:
        print('WRONG... ',MSE_diff)

#Reset weights
NN_test.Theta_flat = Theta_flat

#Train the ANN with sensitivities
MSE_hist = funcs_regression.train_sens_nn(x_train, y_train, dydx_train, NN_test, opt_options)[0]

#Use feedforward
y_out_train_sens = NN_test.feedforward(x_train)[0]
y_out_test_sens = NN_test.feedforward(x_test)[0]

#Compare final MSE
MSE_diff = np.abs(MSE_hist[-1] - 0.0010439480345924856) #5-5 neurons, 5-5 samples, no sens

print('Check Training with sensitivities:')
if MSE_diff < 1e-13:
        print('OK! ',MSE_diff)
else:
        print('WRONG... ',MSE_diff)
############################################

############################################
#FINISH PRINT STATEMENTS
print('')
print('           REGRESSION TEST ENDED            ')
print('############################################')
############################################

############################################
#PLOT RESULTS
#More imports
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
#Prepare data to plot
scatter_targets = np.vstack([x_train, y_train])
scatter_ANN_nosens = np.vstack([x_train, y_out_train_nosens])
scatter_ANN_sens = np.vstack([x_train, y_out_train_sens])
Y_out_test_nosens = y_out_test_nosens.reshape(Y_test.shape)
Y_out_test_sens = y_out_test_sens.reshape(Y_test.shape)

#Plotting results without sensitivities
fig = plt.figure()
ax = fig.gca(projection='3d')
surf1 = ax.plot_surface(X1_test, X2_test, Y_test, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#surf1 = ax.plot_wireframe(X1_test, X2_test, Y_test, rstride=1, cstride=1)
#surf2 = ax.plot_surface(X1_test, X2_test, Y_out, rstride=1, cstride=1, linewidth=0, antialiased=False)
surf2 = ax.plot_wireframe(X1_test, X2_test, Y_out_test_nosens, rstride=1, cstride=1)
ax.scatter(scatter_targets[0,:],scatter_targets[1,:],scatter_targets[2,:], c='r', marker='o')
ax.scatter(scatter_ANN_nosens[0,:],scatter_ANN_nosens[1,:],scatter_ANN_nosens[2,:], c='b', marker='o')

#Plotting results with sensitivities
fig = plt.figure()
ax = fig.gca(projection='3d')
surf1 = ax.plot_surface(X1_test, X2_test, Y_test, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#surf1 = ax.plot_wireframe(X1_test, X2_test, Y_test, rstride=1, cstride=1)
#surf2 = ax.plot_surface(X1_test, X2_test, Y_out, rstride=1, cstride=1, linewidth=0, antialiased=False)
surf2 = ax.plot_wireframe(X1_test, X2_test, Y_out_test_sens, rstride=1, cstride=1)
ax.scatter(scatter_targets[0,:],scatter_targets[1,:],scatter_targets[2,:], c='r', marker='o')
ax.scatter(scatter_ANN_sens[0,:],scatter_ANN_sens[1,:],scatter_ANN_sens[2,:], c='b', marker='o')

plt.show()
############################################
