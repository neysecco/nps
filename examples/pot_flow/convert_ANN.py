#import sys
#sys.path.append('../../')
import sys
import nps
import pickle

canonical_prob   = ['potflow_singlenet', 'potflow_doublenet']

def eval_case(canonical_prob):

    #sys.modules['NPS'] = nps

    ###########################################
    with open('./' + canonical_prob + '/results.pickle','rb') as fid:
        results = pickle.load(fid)

    print(results.keys())

    NN_set = []

    for NN_old in results['NN_set']:

        NN_new = nps.NN(num_inputs=NN_old.num_inputs,
                        num_outputs=NN_old.num_outputs,
                        num_neurons_list=NN_old.num_neurons_list,
                        layer_type=['sigmoid']*len(NN_old.num_neurons_list),
                        Theta_flat=NN_old.Theta_flat,
                        lower_bounds=NN_old.lower_bounds[:,0],
                        upper_bounds=NN_old.upper_bounds[:,0])

        NN_set.append(NN_new)


    results['NN_set'] = NN_set

    with open('./' + canonical_prob + '/results2.pickle','wb') as fid:
        pickle.dump(results, fid)

    ###########################################
    #NN_set = results['NN_set']
    #Theta_flat_hist      = results['Theta_flat_hist']
    #Theta_flat_ALM_hist  = results['Theta_flat_ALM_hist']
        

#############################################

for i in range(len(canonical_prob)):
    eval_case(canonical_prob[i])
