# GENERAL IMPORTS
import numpy as np

###########################################

class NN:

    #Initialization function
    def __init__(self, num_inputs, num_outputs, num_neurons_list=[20,20],
                 layer_type = None, Theta_flat = None, lower_bounds = None, upper_bounds = None):
        '''
        This is the main gateway to create an Artificial Neural Network (ANN) with
        Extended Backpropagation capabilities.
        
        INPUTS:
        num_inputs: integer -> Dimension of input vector (not number of cases).
        num_outputs: integer -> Number of outputs of the ANN.
        num_neurons_list: 1D array of integers -> Number of neurons in each hidden layer.
        layer_type: 1D array of strings -> Strings that indicate the type of each hidden layer.
                                           The supported types are 'sigmoid' and 'linear'.
        Theta_flat: 1D array of doubles -> Array that can be used to initialize the ANN weights.
                                           If None, random weights will be assigned using
                                           the Nguyen-Widrow algorithm.
                                           If an integer, this is used as seed to always
                                           generate the same random numbers.
        lower_bound: 1D array of doubles -> Expected lower bounds for each
                                            input variable used for normalization
        upper_bound: 1D array of doubles -> Expected upper bounds for each
                                            input variable used for normalization
        '''

        # Store the number of inputs and outputs
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # Store an array that contains the number of neurons per layer
        self.num_neurons_list = np.array(num_neurons_list)

        # Store number of hidden layers
        self.num_hidden_layers = len(num_neurons_list)

        ## TRANSFER FUNCTIONS

        # Set hidden layers as sigmoid if user did not provide any
        if layer_type is None:
            self.layer_type = self.num_hidden_layers*['sigmoid']
        
        # Otherwise check if the user gave the right number of types
        elif len(layer_type) == self.num_hidden_layers:
            self.layer_type = layer_type # Assing the user-provided types

        else: #Print error message
            raise ValueError('User did not provide the correct number of layer types')

        ## INITIALIZATION OF WEIGHTS

        # Compute the total number of weights
        self.num_theta = (num_inputs+1)*self.num_neurons_list[0] + \
                sum((self.num_neurons_list[:-1]+1)*self.num_neurons_list[1:]) + \
                (self.num_neurons_list[-1]+1)*num_outputs 

        if type(Theta_flat) in [int, type(None)]: # Check if the user requires randomly generated weights
            
            # Check if the user provided a seed for the random number generation
            if type(Theta_flat) is int:
                rng = np.random.default_rng(seed=Theta_flat)
            else:
                rng = np.random.default_rng()
                
            #Assign random weights with the Nguyen-Widrow initialization
            self.Theta_flat = weights_init(num_inputs, num_outputs, num_neurons_list, self.num_theta, rng)

        elif len(Theta_flat) == self.num_theta: #Check if user provided the correct number of weights
            
            #Assign the user-provided weights
            self.Theta_flat = Theta_flat

        else: #Print error message
            raise ValueError('User did not provide the correct number of weights')

        ## NORMALIZATION OPTIONS

        # Set up normalization data and transformation matrices
        if (lower_bounds is None) or (upper_bounds is None):
            self.lower_bounds = -1.0*np.ones((num_inputs,1))
            self.upper_bounds = +1.0*np.ones((num_inputs,1))
        else:
            self.lower_bounds = np.array([lower_bounds]).T
            self.upper_bounds = np.array([upper_bounds]).T

        # Matrix to normalize first order derivatives
        self.normT = 2.0/(self.upper_bounds-self.lower_bounds)

        # Matrix to normalize second order derivatives
        self.normT2 = np.hstack([self.normT]*self.num_inputs)*np.hstack([self.normT]*self.num_inputs).T

    ###########################################

    def feedforward(self, Inputs_raw):
        '''
        INPUTS
        Inputs_raw: float[num_inputs, num_cases] -> Array of sampling points
                    without normalization.
        '''

        #IMPORTS
        from .nn_fortran import feedforward

        # Normalize inputs
        Inputs = 2.0*(Inputs_raw-self.lower_bounds)/(self.upper_bounds-self.lower_bounds) - 1.0

        # Mapping layer types to integers according to what is defined in the Fortran code
        hidLayerTypes = map_layer_type(self.layer_type)

        # Call the Fortran backpropagation code
        [Outputs,
         dOutputs_dInputs,
         dOutputs_dInputsdInputs] = feedforward(self.Theta_flat, 
                                                hidLayerTypes,
                                                self.num_neurons_list,
                                                Inputs)

        # Denormalize
        dOutputs_dInputs = dOutputs_dInputs*self.normT
        dOutputs_dInputsdInputs = dOutputs_dInputsdInputs*self.normT2[:,:,None]

        #Returning the Outputs values and hidden layers activations
        return Outputs, dOutputs_dInputs, dOutputs_dInputsdInputs
        #Outputs[k] = y_k
        #dOutputs_dInputs[j,k] = dy_k/dx_jk
        #dOutputs_dInputsdInputs[i,j,k] = dy_k/dx_ik/dx_jk

    ###########################################

    def backpropagation(self,Inputs_raw):
        '''
        INPUTS
        Inputs_raw: float[num_inputs, num_cases] -> Array of sampling points
                    without normalization.
        '''

        #IMPORTS
        from .nn_fortran import backpropagation

        # Normalize inputs
        Inputs = 2.0*(Inputs_raw-self.lower_bounds)/(self.upper_bounds-self.lower_bounds) - 1.0

        # Mapping layer types to integers according to what is defined in the Fortran code
        hidLayerTypes = map_layer_type(self.layer_type)

        # Call the Fortran backpropagation code
        [Outputs,
         dOutputs_dInputs,
         dOutputs_dInputsdInputs,
         dOutputs_dTheta,
         dSens_dTheta,
         dHess_dTheta] = backpropagation(self.Theta_flat, hidLayerTypes, self.num_neurons_list, Inputs)

        # Denormalize
        dOutputs_dInputs = dOutputs_dInputs*self.normT
        dOutputs_dInputsdInputs = dOutputs_dInputsdInputs*self.normT2[:,:,None]
        dSens_dTheta = dSens_dTheta*self.normT[None,:,:]
        dHess_dTheta = dHess_dTheta*self.normT2[None,:,:,None]

        return dOutputs_dTheta, dSens_dTheta, dHess_dTheta, Outputs, dOutputs_dInputs, dOutputs_dInputsdInputs
        #dOutputs_dTheta[p*q, k] = dy_k/dtheta_pq
        #dSens_dTheta[p*q, j, k] = d(dy_k/dx_jk)/dtheta_pq
        #dHess_dTheta[p*q, i, j, k] = d(dy_k/dx_jk/dx_ik)/dtheta_pq
        #Outputs[k] = y_k
        #dOutputs_dInputs[j,k] = dy_k/dx_jk
        #dOutputs_dInputsdInputs[i,j,k] = dy_k/dx_ik/dx_jk
        #OBS: dOutputs_dInputs should be equal to dOutputs_dInputs provided by the feedforward function.
        #     This is checked during the regression test.
        #OBS: dOutputs_dInputs = Sens
        #OBS: dOutputs_dInputsdInputs = Hess

    ###########################################

#END OF NN CLASS
###########################################
#AUXILIARY FUNCTIONS

###########################################

def map_layer_type(layer_type):
    '''
    This functions converts the strings in the layer_type list
    into equivalent integers that are coded in Fortran
    '''
    
    layerTypeDict = {'linear': 1,
                     'sigmoid': 2}
    hidLayerTypes = [layerTypeDict[s] for s in layer_type]

    return hidLayerTypes

###########################################

###########################################

def extract_theta(NN_obj, layer_index):
    '''
    This function extracts the weight matrix for a specific layer from the flat weight vector
    layer_index=0 Return Theta between input and first hidden layer
    '''

    # Expand the list of neurons to include the number of inputs and outputs
    num_neurons_list = np.hstack([[NN_obj.num_inputs], NN_obj.num_neurons_list, [NN_obj.num_outputs]])

    # Find the start index to slice the weight array
    start_index = 0

    for current_layer in range(layer_index): # This loop will execute until the desired layer is reached
        start_index = start_index + (num_neurons_list[current_layer]+1)*num_neurons_list[current_layer+1]

    # Find the end index
    end_index = start_index + (num_neurons_list[layer_index]+1)*num_neurons_list[layer_index+1]

    # Slice the flat weight array
    Theta = NN_obj.Theta_flat[start_index:end_index]

    # Reshape it to matrix form
    Theta = Theta.reshape((num_neurons_list[layer_index]+1, num_neurons_list[layer_index+1])).T

    # RETURNS
    return Theta

###########################################

###########################################

def weights_init(num_inputs, num_outputs, num_neurons_list, num_theta, rng):
    '''
    This function initializes the ANN weights using the Nguyen-Widrow
    initialization algorithm
    '''

    # Preallocate flat weight array
    Theta_flat = np.zeros(num_theta)

    # Set counter that will help slice the full Theta_flat array
    position = 0

    ## INPUT LAYER

    # Get NW weights
    Theta_flat_part = NW_init(num_inputs, num_neurons_list[0], rng)

    # Add weights to the full array
    Theta_flat[position:position+len(Theta_flat_part)] = Theta_flat_part

    # Increment slicing counter
    position = position + len(Theta_flat_part)

    ## HIDDEN LAYERS
    for index in range(1,len(num_neurons_list)):

        # Get NW weights
        Theta_flat_part = NW_init(num_neurons_list[index-1],num_neurons_list[index],rng)

        # Add weights to the full array
        Theta_flat[position:position+len(Theta_flat_part)] = Theta_flat_part

        # Increment slicing counter
        position = position + len(Theta_flat_part)

    ## OUTPUT LAYER

    # Get NW weights
    Theta_flat_part = NW_init(num_neurons_list[-1], num_outputs, rng)

    # Add weights to the full array
    Theta_flat[position:position+len(Theta_flat_part)] = Theta_flat_part

    # Increment slicing counter
    position = position + len(Theta_flat_part)

    # RETURNS
    return Theta_flat

###########################################

###########################################

def NW_init(num_neurons_prev, num_neurons_next, rng):
    '''
    This auxiliary function contains the
    basic expressions of the Nguyen-Widrow weights initialization.
    num_neurons_prev is the number of neurons of the previous layer (l-1).
    num_neurons_next is the number of neurons of the next layer (l).
    This function gives Theta for layer (l).
    '''

    # Compute desired weight vector magnitude
    magW = num_neurons_next**(1/num_neurons_prev)

    # Generate the biases
    Wbias = (2*rng.random((num_neurons_next,1))-1)*magW

    # Generate other weights
    Wneu = 2*rng.random((num_neurons_next,num_neurons_prev))-1

    # Normalize each row by the desired magnitude
    Wneu_norm = np.array([np.linalg.norm(Wneu, axis=1)]).T # We need a column vector
    Wneu = Wneu/Wneu_norm*0.7*magW

    # Now assemble all weights
    W = np.hstack([Wbias, Wneu])

    # Flatten the weights
    Theta_flat = W.T.ravel()

    # RETURNS
    return Theta_flat

###########################################
