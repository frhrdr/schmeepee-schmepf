#!/bin/bash

module load cuda/11.3


case $1 in
0 )
python3.9 dp_mepf.py --n_iter 20_000 --restart_iter 10_000 --ckpt_iter 10_000 --syn_eval_iter 5_000 --dataset cifar10 --val_enc fid_features --val_data train --exp_name 02_cifar_fid_debug/run_2_tanh --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 3e-4 --seed 1 --gen_output tanh --data_scale 0_1 --extra_input_scaling imagenet_norm
;; 1 )
python3.9 dp_mepf.py --n_iter 20_000 --restart_iter 10_000 --ckpt_iter 10_000 --syn_eval_iter 5_000 --dataset cifar10 --val_enc fid_features --val_data train --exp_name 02_cifar_fid_debug/run_3_tanh --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 3e-4 --seed 1 --gen_output tanh --data_scale 0_1 --extra_input_scaling dataset_norm
;; esac

CODE=$?
if [ $CODE -eq 0 ]
then
curl -H ta:partying_face -d "cifar10_fid_debug run $1 done." ntfy.sh/frederik_is_cluster_alerts_yeah
elif [ $CODE -ne 3 ]
then
curl -H ta:lady_beetle -d "cifar10_fid_debug run $1 failed. code:$CODE." ntfy.sh/frederik_is_cluster_alerts_ohno
fi
exit $CODE
