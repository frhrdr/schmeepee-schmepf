#!/bin/bash

module load cuda/11.3

case $1 in
0 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_0 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 1
;; 1 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_1 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 2
;; 2 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_2 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 3
;; 3 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_3 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 4
;; 4 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_4 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 5
;; 5 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_5 --dp_tgt_eps 0.2 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 1
;; 6 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_6 --dp_tgt_eps 0.2 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 2
;; 7 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_7 --dp_tgt_eps 0.2 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 3
;; 8 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_8 --dp_tgt_eps 0.2 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 4
;; 9 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_9 --dp_tgt_eps 0.2 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 5
;; 10 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_10 --dp_tgt_eps 0.5 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 1
;; 11 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_11 --dp_tgt_eps 0.5 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 2
;; 12 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_12 --dp_tgt_eps 0.5 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 3
;; 13 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_13 --dp_tgt_eps 0.5 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 4
;; 14 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_14 --dp_tgt_eps 0.5 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 5
;; 15 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_15 --dp_tgt_eps 0.5 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 1
;; 16 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_16 --dp_tgt_eps 0.5 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 2
;; 17 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_17 --dp_tgt_eps 0.5 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 3
;; 18 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_18 --dp_tgt_eps 0.5 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 4
;; 19 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_19 --dp_tgt_eps 0.5 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 5
;; 20 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_20 --dp_tgt_eps 1. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 1
;; 21 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_21 --dp_tgt_eps 1. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 2
;; 22 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_22 --dp_tgt_eps 1. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 3
;; 23 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_23 --dp_tgt_eps 1. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 4
;; 24 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_24 --dp_tgt_eps 1. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 5
;; 25 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_25 --dp_tgt_eps 1. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 1
;; 26 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_26 --dp_tgt_eps 1. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 2
;; 27 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_27 --dp_tgt_eps 1. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 3
;; 28 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_28 --dp_tgt_eps 1. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 4
;; 29 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_29 --dp_tgt_eps 1. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 5
;; 30 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_30 --dp_tgt_eps 2. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 1
;; 31 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_31 --dp_tgt_eps 2. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 2
;; 32 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_32 --dp_tgt_eps 2. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 3
;; 33 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_33 --dp_tgt_eps 2. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 4
;; 34 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_34 --dp_tgt_eps 2. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 5
;; 35 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_35 --dp_tgt_eps 2. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 1
;; 36 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_36 --dp_tgt_eps 2. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 2
;; 37 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_37 --dp_tgt_eps 2. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 3
;; 38 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_38 --dp_tgt_eps 2. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 4
;; 39 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_39 --dp_tgt_eps 2. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 5
;; 40 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_40 --dp_tgt_eps 5. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 1
;; 41 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_41 --dp_tgt_eps 5. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 2
;; 42 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_42 --dp_tgt_eps 5. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 3
;; 43 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_43 --dp_tgt_eps 5. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 4
;; 44 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_44 --dp_tgt_eps 5. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 5
;; 45 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_45 --dp_tgt_eps 5. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 1
;; 46 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_46 --dp_tgt_eps 5. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 2
;; 47 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_47 --dp_tgt_eps 5. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 3
;; 48 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_48 --dp_tgt_eps 5. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 4
;; 49 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_49 --dp_tgt_eps 5. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 5
;; 50 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_50 --dp_tgt_eps 10. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 1
;; 51 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_51 --dp_tgt_eps 10. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 2
;; 52 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_52 --dp_tgt_eps 10. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 3
;; 53 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_53 --dp_tgt_eps 10. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 4
;; 54 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_54 --dp_tgt_eps 10. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 5
;; 55 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_55 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 1
;; 56 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_56 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 2
;; 57 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_57 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 3
;; 58 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_58 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 4
;; 59 )
python3.9 dp_mepf.py --n_iter 200_000  --ckpt_iter 25_000 --restart_iter 25_000 --syn_eval_iter 5_000 --dataset celeba --image_size 64 --val_enc fid_features --val_data train --exp_name oct10_celeba_64x64_prelim_results/run_59 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-4 --m_avg_lr 3e-4 --seed 5
;; esac

CODE=$?
if [ $CODE -eq 0 ]
then
curl -H ta:partying_face -d "oct10_celeba_64x64_prelim_results run $1 done." ntfy.sh/frederik_is_cluster_alerts_yeah
elif [ $CODE -ne 3 ]
then
curl -H ta:lady_beetle -d "oct10_celeba_64x64_prelim_results run $1 failed. code:$CODE." ntfy.sh/frederik_is_cluster_alerts_ohno
fi
exit $CODE
