#!/bin/bash

module load cuda/11.3

case $1 in
0 )
python3.9 dp_mepf.py --n_iter 200_000 --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --val_enc fid_features --val_data train --exp_name oct12_celeba_prelim_results_nondp/run_0 --dp_tgt_eps 1000. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-4 --seed 1
;; 1 )
python3.9 dp_mepf.py --n_iter 200_000 --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --val_enc fid_features --val_data train --exp_name oct12_celeba_prelim_results_nondp/run_1 --dp_tgt_eps 1000. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-4 --seed 2
;; 2 )
python3.9 dp_mepf.py --n_iter 200_000 --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --val_enc fid_features --val_data train --exp_name oct12_celeba_prelim_results_nondp/run_2 --dp_tgt_eps 1000. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-4 --seed 3
;; 3 )
python3.9 dp_mepf.py --n_iter 200_000 --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --val_enc fid_features --val_data train --exp_name oct12_celeba_prelim_results_nondp/run_3 --dp_tgt_eps 1000. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-4 --seed 4
;; 4 )
python3.9 dp_mepf.py --n_iter 200_000 --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --val_enc fid_features --val_data train --exp_name oct12_celeba_prelim_results_nondp/run_4 --dp_tgt_eps 1000. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-4 --seed 5
;; 5 )
python3.9 dp_mepf.py --n_iter 200_000 --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --val_enc fid_features --val_data train --exp_name oct12_celeba_prelim_results_nondp/run_5 --dp_tgt_eps 1000. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-4 --seed 1
;; 6 )
python3.9 dp_mepf.py --n_iter 200_000 --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --val_enc fid_features --val_data train --exp_name oct12_celeba_prelim_results_nondp/run_6 --dp_tgt_eps 1000. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-4 --seed 2
;; 7 )
python3.9 dp_mepf.py --n_iter 200_000 --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --val_enc fid_features --val_data train --exp_name oct12_celeba_prelim_results_nondp/run_7 --dp_tgt_eps 1000. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-4 --seed 3
;; 8 )
python3.9 dp_mepf.py --n_iter 200_000 --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --val_enc fid_features --val_data train --exp_name oct12_celeba_prelim_results_nondp/run_8 --dp_tgt_eps 1000. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-4 --seed 4
;; 9 )
python3.9 dp_mepf.py --n_iter 200_000 --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --val_enc fid_features --val_data train --exp_name oct12_celeba_prelim_results_nondp/run_9 --dp_tgt_eps 1000. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-4 --seed 5
;; esac

CODE=$?
if [ $CODE -eq 0 ]
then
curl -H ta:partying_face -d "oct12_celeba_prelim_results_nondp run $1 done." ntfy.sh/frederik_is_cluster_alerts_yeah
elif [ $CODE -ne 3 ]
then
curl -H ta:lady_beetle -d "oct12_celeba_prelim_results_nondp run $1 failed. code:$CODE." ntfy.sh/frederik_is_cluster_alerts_ohno
fi
exit $CODE
