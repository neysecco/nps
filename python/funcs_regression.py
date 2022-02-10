#GENERAL IMPORTS
from __future__ import division

def train_nn(Inputs, Targets, NN, opt_options={}, check_gradients=False):
        #This function uses the backpropagation algorithm to train the ANN. The backpropagation is performed with a wrapped Fortran code
        #INPUTS:
        #Theta_flat0: 1 x num_theta array: initial set of weights

        #IMPORTS
        from .optimizers import fminscg
        from .num_diff import center_fd
        from numpy import array

        # Get initial set of weights
        Theta_flat0 = NN.Theta_flat.copy()

        #Define objective function
        def get_MSE_gradients(Theta_flat):
                #Update weights
                NN.Theta_flat = Theta_flat
                #Use backpropagation
                MSE, dMSE_dTheta = MSE_values(Inputs,Targets,NN)
                #RETURNS
                return MSE, dMSE_dTheta

        if check_gradients: #Check if the user wants to verify the gradients
                #Define simplified function that returns only residuals, in the format required by num_diff.py
                def test(Theta_flat_col):

                        #Update weights
                        NN.Theta_flat = Theta_flat_col

                        # Get MSE
                        MSE = MSE_values(Inputs, Targets, NN)[0]

                        # RETURNS
                        return array([[MSE]])

                #Try first point
                dMSE = get_MSE_gradients(Theta_flat0)[1:] #Get only the gradients
                dMSE_num = center_fd(test,array([Theta_flat0]).T,1e-6)[:,:,0]

                #Check the difference
                dMSE_diff = max(max(abs(dMSE-dMSE_num)))

                #Print log
                print('funcs_regression.train_nn: Gradient verification requested...')
                print('Maximum absolute difference for MSE function: ',dMSE_diff)
                if dMSE_diff < 1e-6:
                        print('OK! Residual functions are correct!')
                else:
                        print('WRONG! Verify MSE gradients!')
                        exit()

        #Run the optimizer
        Theta_flat_hist,MSE_hist,motive = fminscg(get_MSE_gradients, NN.Theta_flat, **opt_options)

        #Get the final answer
        Theta_flatf = Theta_flat_hist[:,-1]

        #Update the network set of weights
        NN.Theta_flat = Theta_flatf

        #RETURNS
        return MSE_hist, Theta_flat_hist
        #This function also updates:
        #NN.Theta_flat

###########################################

def train_sens_nn(Inputs, Targets, Sens_Targets, NN, opt_options={}):
        #This function uses the backpropagation algorithm to train the ANN. The backpropagation is performed with a wrapped Fortran code
        #INPUTS:
        #Theta_flat0: 1 x num_theta array: initial set of weights

        #IMPORTS
        from .optimizers import fminscg

        #Define objective function
        def get_nn_gradients(Theta_flat):
                #Update weights
                NN.Theta_flat = Theta_flat
                #Use backpropagation to get the functions gradients
                MSE, dMSE_dTheta = MSE_sensitivities(Inputs,Targets,Sens_Targets,NN)
                #RETURNS
                return MSE, dMSE_dTheta

        #Run the optimizer
        Theta_flat_hist,MSE_hist,motive = fminscg(get_nn_gradients, NN.Theta_flat, **opt_options)

        #Get the final answer
        Theta_flatf = Theta_flat_hist[:,-1]

        #Update the network set of weights
        NN.Theta_flat = Theta_flatf

        #RETURNS
        return MSE_hist, Theta_flat_hist
        #This function also updates:
        #NN.Theta_flat

###########################################

def MSE_values(Inputs,Targets,NN):
        #This function computes the MSE at the given points. It also gives the MSE gradients with respect to the weights

        #IMPORTS
        from numpy import sum

        #Get number of cases
        num_cases = Inputs.shape[1]

        #Get the output gradients with respect to weights
        dOutput_dTheta, dummy, dummy, Outputs = NN.backpropagation(Inputs)[0:4]

        #Compute the MSE
        MSE = sum((Outputs-Targets)**2)/2/num_cases

        #Compute the MSE derivative
        dMSE_dOutputs = (Outputs-Targets)/num_cases
        dMSE_dTheta = dOutput_dTheta.dot(dMSE_dOutputs.T)[:,0]

        return MSE, dMSE_dTheta

###########################################

def MSE_sensitivities(Inputs,Targets,Sens_Targets,NN):
        #This function computes the MSE at the given points. It also gives the MSE gradients with respect to the weights

        #IMPORTS
        from numpy import sum, array

        #Get number of cases
        num_cases = Inputs.shape[1]

        #Get the output and sensitivities gradients with respect to weights
        dOutputs_dTheta, dSens_dTheta, dummy, Outputs, Sens = NN.backpropagation(Inputs)[0:5]

        #Compute the MSE
        MSE = sum((Outputs-Targets)**2)/2/num_cases + sum((Sens-Sens_Targets)**2)/2/num_cases

        #Compute the MSE derivative
        dMSE_dOutputs = (Outputs-Targets)/num_cases
        dMSE_dTheta_Outputs = dOutputs_dTheta.dot(dMSE_dOutputs.T)[:,0]
        dMSE_dSens = (Sens-Sens_Targets)/num_cases
        dMSE_dTheta_Sens = sum(sum(dSens_dTheta*dMSE_dSens,axis=2),axis=1)

        dMSE_dTheta = dMSE_dTheta_Outputs + dMSE_dTheta_Sens

        return MSE, dMSE_dTheta

###########################################
