# this script is for running multiple runs in series


import sys
import subprocess

seed_num_arr = [ 1, 2, 3, 4, 5 ]
data_arr = ['digits', 'fashion']
epoch_arr = [50,100,200,400]
batch_size_arr = [100, 200, 400]
lr_arr = [5e-5, 1e-3]
loss_type_arr = ['l1', 'l2']
beta1_arr = [0.2, 0.5, 0.7]
m_avg_lr_arr = [1e-5, 0.0002, 1e-3]
mean_only_arr = ['True', 'False']
feat_selection_prec_arr = [10, 20, 50, 80, 100]


for data_name in data_arr:
    sys.argv()
    execfile('train_gen_moving_avg.py')