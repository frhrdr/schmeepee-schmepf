#!/bin/bash

module load cuda/11.3

case $1 in
0 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_0 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 1
;; 1 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_1 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 2
;; esac

CODE=$?
if [ $CODE -eq 0 ]
then
curl -H ta:partying_face -d "oct10_celeba_test run $1 done." ntfy.sh/frederik_is_cluster_alerts_yeah
elif [ $CODE -ne 3 ]
then
curl -H ta:lady_beetle -d "oct10_celeba_test run $1 failed. code:$CODE." ntfy.sh/frederik_is_cluster_alerts_ohno
fi
exit $CODE
