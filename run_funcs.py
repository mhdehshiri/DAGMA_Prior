import torch
from Dagma import utils
from Dagma.linear import DagmaLinear
from Dagma.nonlinear import DagmaMLP, DagmaNonlinear
import numpy as np


#define model function
def dagma_runner(X , B_prior , d , model_type , sp_coef ,dag_coef , threshold = 0.01):
    if model_type == 'nldagma' :
        eq_model = DagmaMLP(dims=[d, 15, 1],
                    bias=True, dtype=torch.double)
        model = DagmaNonlinear(eq_model, dtype=torch.double) 
        W_est = model.fit(X, lambda1=0.001 , w_threshold = threshold) 
    else :
        model = DagmaLinear(loss_type='l2')
        W_est = model.fit(X, B_pri = B_prior ,  lambda1= sp_coef , w_threshold = threshold) 
    return W_est


# def dagma_runner(dataset, sparsity_coef_list, dag_coef_list, sub_nums , num_samples 
#                  ,model_type , ul , data_folder , data_name):
    