#!/bin/bash

module load cuda/11.3

case $1 in
0 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_0 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 1 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_1 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 2 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_2 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 3 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_3 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 4 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_4 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 5 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_5 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 6 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_6 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 7 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_7 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 8 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_8 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 9 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_9 --dp_tgt_eps 0.2 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 10 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_10 --dp_tgt_eps 0.2 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 11 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_11 --dp_tgt_eps 0.2 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 12 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_12 --dp_tgt_eps 0.2 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 13 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_13 --dp_tgt_eps 0.2 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 14 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_14 --dp_tgt_eps 0.2 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 15 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_15 --dp_tgt_eps 0.2 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 16 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_16 --dp_tgt_eps 0.2 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 17 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_17 --dp_tgt_eps 0.2 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 18 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_18 --dp_tgt_eps 0.2 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 19 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_19 --dp_tgt_eps 0.2 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 20 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_20 --dp_tgt_eps 0.5 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 21 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_21 --dp_tgt_eps 0.5 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 22 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_22 --dp_tgt_eps 0.5 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 23 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_23 --dp_tgt_eps 0.5 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 24 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_24 --dp_tgt_eps 0.5 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 25 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_25 --dp_tgt_eps 0.5 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 26 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_26 --dp_tgt_eps 0.5 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 27 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_27 --dp_tgt_eps 0.5 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 28 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_28 --dp_tgt_eps 0.5 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 29 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_29 --dp_tgt_eps 0.5 --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 30 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_30 --dp_tgt_eps 0.5 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 31 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_31 --dp_tgt_eps 0.5 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 32 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_32 --dp_tgt_eps 0.5 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 33 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_33 --dp_tgt_eps 0.5 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 34 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_34 --dp_tgt_eps 0.5 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 35 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_35 --dp_tgt_eps 0.5 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 36 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_36 --dp_tgt_eps 0.5 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 37 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_37 --dp_tgt_eps 0.5 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 38 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_38 --dp_tgt_eps 0.5 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 39 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_39 --dp_tgt_eps 0.5 --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 40 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_40 --dp_tgt_eps 1. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 41 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_41 --dp_tgt_eps 1. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 42 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_42 --dp_tgt_eps 1. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 43 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_43 --dp_tgt_eps 1. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 44 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_44 --dp_tgt_eps 1. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 45 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_45 --dp_tgt_eps 1. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 46 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_46 --dp_tgt_eps 1. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 47 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_47 --dp_tgt_eps 1. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 48 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_48 --dp_tgt_eps 1. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 49 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_49 --dp_tgt_eps 1. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 50 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_50 --dp_tgt_eps 1. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 51 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_51 --dp_tgt_eps 1. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 52 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_52 --dp_tgt_eps 1. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 53 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_53 --dp_tgt_eps 1. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 54 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_54 --dp_tgt_eps 1. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 55 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_55 --dp_tgt_eps 1. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 56 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_56 --dp_tgt_eps 1. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 57 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_57 --dp_tgt_eps 1. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 58 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_58 --dp_tgt_eps 1. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 59 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_59 --dp_tgt_eps 1. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 60 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_60 --dp_tgt_eps 2. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 61 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_61 --dp_tgt_eps 2. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 62 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_62 --dp_tgt_eps 2. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 63 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_63 --dp_tgt_eps 2. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 64 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_64 --dp_tgt_eps 2. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 65 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_65 --dp_tgt_eps 2. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 66 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_66 --dp_tgt_eps 2. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 67 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_67 --dp_tgt_eps 2. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 68 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_68 --dp_tgt_eps 2. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 69 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_69 --dp_tgt_eps 2. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 70 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_70 --dp_tgt_eps 2. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 71 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_71 --dp_tgt_eps 2. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 72 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_72 --dp_tgt_eps 2. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 73 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_73 --dp_tgt_eps 2. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 74 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_74 --dp_tgt_eps 2. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 75 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_75 --dp_tgt_eps 2. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 76 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_76 --dp_tgt_eps 2. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 77 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_77 --dp_tgt_eps 2. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 78 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_78 --dp_tgt_eps 2. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 79 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_79 --dp_tgt_eps 2. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 80 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_80 --dp_tgt_eps 5. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 81 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_81 --dp_tgt_eps 5. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 82 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_82 --dp_tgt_eps 5. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 83 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_83 --dp_tgt_eps 5. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 84 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_84 --dp_tgt_eps 5. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 85 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_85 --dp_tgt_eps 5. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 86 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_86 --dp_tgt_eps 5. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 87 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_87 --dp_tgt_eps 5. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 88 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_88 --dp_tgt_eps 5. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 89 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_89 --dp_tgt_eps 5. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 90 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_90 --dp_tgt_eps 5. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 91 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_91 --dp_tgt_eps 5. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 92 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_92 --dp_tgt_eps 5. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 93 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_93 --dp_tgt_eps 5. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 94 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_94 --dp_tgt_eps 5. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 95 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_95 --dp_tgt_eps 5. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 96 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_96 --dp_tgt_eps 5. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 97 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_97 --dp_tgt_eps 5. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 98 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_98 --dp_tgt_eps 5. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 99 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_99 --dp_tgt_eps 5. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 100 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_100 --dp_tgt_eps 10. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 101 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_101 --dp_tgt_eps 10. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 102 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_102 --dp_tgt_eps 10. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 103 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_103 --dp_tgt_eps 10. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 104 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_104 --dp_tgt_eps 10. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 105 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_105 --dp_tgt_eps 10. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 106 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_106 --dp_tgt_eps 10. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 107 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_107 --dp_tgt_eps 10. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 108 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_108 --dp_tgt_eps 10. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 109 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_109 --dp_tgt_eps 10. --matched_moments m1_and_m2 --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 110 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_110 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 111 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_111 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 112 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_112 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 113 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_113 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 114 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_114 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 64 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; 115 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_115 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 1
;; 116 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_116 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 2
;; 117 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_117 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 3
;; 118 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_118 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 4
;; 119 )
python3 dp_mepf.py --n_iter 200_000 --restart_iter 25_000 --ckpt_iter 25_000 --syn_eval_iter 5_000 --dataset cifar10 --labeled --val_enc fid_features --val_data train --exp_name oct11_cifar10_labeled_prelim_results/run_119 --dp_tgt_eps 10. --matched_moments mean --dp_val_noise_scaling 10. --batch_size 128 --lr 1e-3 --m_avg_lr 1e-2 --seed 5
;; esac

CODE=$?
if [ $CODE -eq 0 ]
then
curl -H ta:partying_face -d "oct11_cifar10_labeled_prelim_results run $1 done." ntfy.sh/frederik_is_cluster_alerts_yeah
elif [ $CODE -ne 3 ]
then
curl -H ta:lady_beetle -d "oct11_cifar10_labeled_prelim_results run $1 failed. code:$CODE." ntfy.sh/frederik_is_cluster_alerts_ohno
fi
exit $CODE
