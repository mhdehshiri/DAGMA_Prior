import torch
from dagma import utils
from dagma.linear import DagmaLinear
from dagma.nonlinear import DagmaMLP, DagmaNonlinear
import numpy as np
from auxiliary_funcs import *


#dataset and loading and saving adds 
data_folder = 'Data/'  
results_folder = 'outcomes/'
data_name = 'mean1_dataset'
# data_name = 'mean2_dataset'
ul = '_'
dataset = np.load(data_folder + data_name + '.npy')
print("********"," dataset name is " , data_name , " ********")
#prepare the hyperparameters list
sparsity_coef_list = [0.0005 , 0.001 , 0.005 , 0.01]
# sparsity_coef_list = [0.0005]
dag_coef_list = [0.005]
sub_nums = 20
node_nums = 164
num_samples = 1200
model_type = 'ldagma'
#model and training
for dag_id , dag_coef in enumerate(dag_coef_list) :
    for sp_id , sp_coef in enumerate(sparsity_coef_list):
        X = dataset[:,:num_samples,0]
        d = X.shape[1]
        eq_model = DagmaMLP(dims=[d, 15, 1],
                    bias=True, dtype=torch.double)
        save_name = [data_name + ul + model_type + ul + 'num_subs_' + 
                    str(sub_nums) + ul + 'sparsity_' + str(sp_coef) +
                    ul + 'dagness' + ul + str(dag_coef) ][0]
        add_strc = data_folder + "s_c_final_80.npy"

        connectivity_map(add1 = add_strc , add2 = results_folder+ save_name ,
                        edges = int(1444) , name1 = 'Structural' , name2 = 'Estimated' , sub_id=10)