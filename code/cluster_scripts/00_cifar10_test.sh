#!/bin/bash

module load cuda/11.3

case $1 in
0 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 5_000 --syn_eval_iter 5_000 --dataset cifar10 --val_enc fid_features --val_data train --exp_name cifar10_test/run_0 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 3e-4 --seed 1
;; 1 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 5_000 --syn_eval_iter 5_000 --dataset cifar10 --val_enc fid_features --val_data train --exp_name cifar10_test/run_1 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 3e-4 --seed 5
;; esac

CODE=$?
if [ $CODE -eq 0 ]
then
curl -H ta:partying_face -d "cifar10_test run $1 done." ntfy.sh/frederik_is_cluster_alerts_yeah
elif [ $CODE -ne 3 ]
then
curl -H ta:lady_beetle -d "cifar10_test run $1 failed. code:$CODE." ntfy.sh/frederik_is_cluster_alerts_ohno
fi
exit $CODE
