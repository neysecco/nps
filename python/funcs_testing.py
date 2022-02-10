#This set of functions is used by regression_test.py for the regression test (durrr!)
#This file contains derivatives computed with numerical methods, so we can compare with analytical ones

###########################################

def test_num_backprop(Inputs,Targets,Sens_Targets,NN):

        #This function computes the sensitivites of MSE for the given set of targets using finite differences. This is useful for the regression test of the entire framework.

        #IMPORTS
        from numpy import array, sum
        from .num_diff import center_fd
        
        #Analysing the problem
        num_cases = Inputs.shape[1] #Number of features and number of cases

        #Unenrolling the weights to form an 1-D array
        Theta_flat_col = array([NN.Theta_flat]).T #complex_step receives variables in a column

        #Defining feedfoward function that receives the weights vector given by the complex_step
        def vector_feedfoward(Theta_flat_col_given):
                #Update weights vector
                NN.Theta_flat = Theta_flat_col_given
                #Now do the usual feedfoward
                Outputs = NN.feedforward(Inputs)[0]
                #Calculating the mean squared error
                MSE = sum((Outputs-Targets)**2)/2/num_cases
                #Returning the mean squared error
                return array([[MSE]]) #The brackets are due to num_diff.complex_step requirements on output formatiing (should be 2D array)

        #Define another function to compute MSE using sensitivities
        def vector_feedfoward_sens(Theta_flat_col_given):
                #Update weights vector
                NN.Theta_flat = Theta_flat_col_given
                #Now do the usual feedfoward
                Outputs, Sens = NN.feedforward(Inputs)[:2]
                #Calculating the mean squared error
                MSE = sum((Outputs-Targets)**2)/2/num_cases
                MSE_sens = sum((Sens-Sens_Targets)**2)/2/num_cases
                #Returning the mean squared error
                return array([[MSE+MSE_sens]]) #The brackets are due to num_diff.complex_step requirements on output formatiing (should be 2D array)

        #Calculating MSE at the given point
        MSE = vector_feedfoward(Theta_flat_col)[0,0]

        #Calculating MSE at the given point with sensitivites
        MSE_sens = vector_feedfoward_sens(Theta_flat_col)[0,0]

        #Calculating the gradient with finite differences
        Grad_flat = center_fd(vector_feedfoward, Theta_flat_col, 1e-7)[0,:,0] #Brackets because num_diff gives 3D array
        Grad_sens_flat = center_fd(vector_feedfoward_sens, Theta_flat_col, 1e-7)[0,:,0] #Brackets because num_diff gives 3D array

        #Change weights vector back to its original value, as the steps may have changed it
        NN.Theta_flat = Theta_flat_col[:,0] #We use the brackets to get a row array instead of a column array

        #RETURNS
        return MSE, Grad_flat, MSE_sens, Grad_sens_flat

###########################################

def test_num_grad(Inputs,NN):
        #This function computes the gradients of the outputs with respect to weights (dOutputs_dTheta) numerically.

        #IMPORTS
        from .num_diff import center_fd
        from numpy import array

        #Unenrolling the weights to form a column array
        Theta_flat_col = array([NN.Theta_flat]).T #center_fd receives variables in a column

        #Due to num_diff requirements, we must rearrange the inputs and outputs of the sensitivities function. Theta_flat_col_given is a column array given by num_diff iteration.
        def feedforward_mod(Theta_flat_col_given):
                #Update the set of weights
                NN.Theta_flat = Theta_flat_col_given
                #Call feedforward function, and get the first output only (sensitivities)
                Outputs = array([NN.feedforward(Inputs)[0]])
                #Now we need to rearrange all these outputs in a single column vector
                Outputs_col = Outputs.T
                #Return
                return Outputs_col

        #Find outputs gradients
        dOutputs_dTheta = center_fd(feedforward_mod,Theta_flat_col,1e-7)[:,:,0].T #The brackets are used to reduce third order tensor to a 2D array. As we computed the gradients for just a single configuration of weights, we just need the element 0 of the case index. See num_diff for details.

        #Change weights vector back to its original value, as the steps may have changed it
        NN.Theta_flat = Theta_flat_col[:,0] #We use the brackets to get a row array instead of a column array

        #RETURNS
        return dOutputs_dTheta

###########################################

def test_num_sensitivities(Inputs,NN):

        #IMPORTS
        from numpy import array
        from .num_diff import center_fd

        #Define a function to take only the first output of the feedforward function
        def ff_output(Inputs):
                return array([NN.feedforward(Inputs)[0]])

        #Calculating the gradient with finite differences
        Gradients = center_fd(ff_output, Inputs, 1e-7)

        return Gradients

###########################################

def test_num_sens_grad(Inputs,NN):

        #IMPORTS
        from .num_diff import center_fd
        from numpy import array

        #Unenrolling the weights to form a column array
        Theta_flat_col = array([NN.Theta_flat]).T #center_fd receives variables in a column

        #Due to num_diff requirements, we must rearrange the inputs and outputs of the sensitivities function. Theta_flat_col_given is a column array given by num_diff iteration.
        def sensitivities_mod(Theta_flat_col_given):
                #Update the set of weights
                NN.Theta_flat = Theta_flat_col_given
                #Call feedforward function, and get the second output only (sensitivities)
                Sens = NN.feedforward(Inputs)[1]
                #Now we need to flatten all these sensitivities in a single column vector
                Sens_flat = array([Sens.ravel()]).T
                #Return
                return Sens_flat #Sens_flat = [[dy1/dx1 dy1/dx2 dy1/dx3 ... dy1/dxn dy2/dx1 dy2/dx2 ... dym/dxn]].T

        #Find sensitivities gradients
        dSens_dTheta = center_fd(sensitivities_mod,Theta_flat_col,1e-7)[:,:,0].T #The brackets are used to reduce third order tensor to a 2D array. As we computed the gradients for just a single configuration of weights, we just need the element 0 of the case index. See num_diff for details.

        #Change weights vector back to its original value, as the steps may have changed it
        NN.Theta_flat = Theta_flat_col[:,0] #We use the brackets to get a row array instead of a column array

        #Reshape
        dSens_dTheta = dSens_dTheta.reshape(-1,Inputs.shape[0],Inputs.shape[1])

        #RETURNS
        return dSens_dTheta

###########################################

def test_num_hessian(Inputs,NN):

        #IMPORTS
        from .num_diff import center_fd
        from numpy import array

        #Define a function to take only the first output of the feedforward function
        def ff_sens(Inputs):
                #Call feedforward to get sensitivities
                return NN.feedforward(Inputs)[1]

        #Calculating the gradient with finite differences
        Gradients = center_fd(ff_sens, Inputs, 1e-7)

        return Gradients

###########################################

def test_num_hess_grad(Inputs,NN):

        #IMPORTS
        from .num_diff import center_fd
        from numpy import array

        #Unenrolling the weights to form a column array
        Theta_flat_col = array([NN.Theta_flat]).T #center_fd receives variables in a column

        #Due to num_diff requirements, we must rearrange the inputs and outputs of the sensitivities function. Theta_flat_col_given is a column array given by num_diff iteration.
        def hessian_mod(Theta_flat_col_given):
                #Update the set of weights
                NN.Theta_flat = Theta_flat_col_given
                #Call feedforward function, and get the second output only (sensitivities)
                Hess = NN.feedforward(Inputs)[2]
                #Now we need to flatten all these sensitivities in a single column vector
                Hess_flat = array([Hess.ravel()]).T
                #Return
                return Hess_flat

        #Find sensitivities gradients
        dHess_dTheta = center_fd(hessian_mod,Theta_flat_col,1e-7)[:,:,0].T #The brackets are used to reduce third order tensor to a 2D array. As we computed the gradients for just a single configuration of weights, we just need the element 0 of the case index. See num_diff for details.

        #Change weights vector back to its original value, as the steps may have changed it
        NN.Theta_flat = Theta_flat_col[:,0] #We use the brackets to get a row array instead of a column array

        #Reshape
        dHess_dTheta = dHess_dTheta.reshape(-1,Inputs.shape[0],Inputs.shape[0],Inputs.shape[1])

        #RETURNS
        return dHess_dTheta

###########################################

#import timeit
#from functools import partial
#time_bp = min(timeit.Timer(partial(NN_test.backpropagation,x_train)).repeat(repeat=3, number=100))
#time_fd = min(timeit.Timer(partial(test_num_sens_grad,x_train)).repeat(repeat=3, number=100))
#print max(abs(dSens_dTheta_bp-dSens_dTheta_cs))
#print 'time_bp:',time_bp
#print 'time_cs:',time_cs
