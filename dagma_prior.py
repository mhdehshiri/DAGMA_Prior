import torch
from Dagma import utils
from Dagma.linear import DagmaLinear
from Dagma.nonlinear import DagmaMLP, DagmaNonlinear
import numpy as np
from auxiliary_funcs import *
from run_funcs import *

#dataset and loading and saving adds 
data_folder = 'Data/'  
results_folder = 'outcomes/'
# data_name = 'mean1_dataset'
data_name = 'mean1_dataset'
prior_name = 'DTI_dataset'
model_type = 'ldagma'
ul = '_'
dataset = np.load(data_folder + data_name + '.npy')
B_PRIOR = -np.exp(np.load(data_folder + prior_name + '.npy' ))
print("********"," dataset name is " , data_name , '  ' , model_type  , " ********")
#prepare the hyperparameters list
sparsity_coef_list = [0.08]
# sparsity_coef_list = [0.0005]
th_coef_list = [0.001]
sub_nums = 62
node_nums = 164
num_samples = 1200
#model and training
for th_id , th_coef in enumerate(th_coef_list) :
    print(f"th id : {th_id}")
    for sp_id , sp_coef in enumerate(sparsity_coef_list):
        print(f"sp_id : {sp_id}")
        W_est = np.zeros((node_nums , node_nums, sub_nums))
        for sub_id in range(0,sub_nums) :
            print(f"sub_id : {sub_id}")
            X = dataset[:num_samples,:node_nums,sub_id]
            B_prior = B_PRIOR[:node_nums,:node_nums,sub_id] 
            d = X.shape[1]
            W_est_t = dagma_runner(X ,B_prior, d , model_type , sp_coef ,th_coef )
            W_est[:,:,sub_id] = W_est_t
            
        save_name = [data_name + ul + prior_name + ul + model_type + ul + 'num_subs_' + 
                     str(sub_nums) + ul + 'sparsity_' + str(sp_coef) +
                     ul + 'thershold' + ul + str(th_coef) ][0]
        np.save(results_folder + save_name , W_est)
     
     