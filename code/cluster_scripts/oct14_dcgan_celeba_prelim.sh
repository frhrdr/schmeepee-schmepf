cp -r ../data/img_align_celeba /tmp/
#!/bin/bash
case $1 in
0 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_0 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 0.2 --seed 1
;; 1 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_1 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 0.2 --seed 2
;; 2 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_2 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 0.2 --seed 3
;; 3 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_3 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 0.2 --seed 4
;; 4 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_4 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 0.2 --seed 5
;; 5 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_5 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 0.5 --seed 1
;; 6 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_6 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 0.5 --seed 2
;; 7 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_7 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 0.5 --seed 3
;; 8 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_8 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 0.5 --seed 4
;; 9 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_9 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 0.5 --seed 5
;; 10 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_10 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 1. --seed 1
;; 11 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_11 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 1. --seed 2
;; 12 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_12 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 1. --seed 3
;; 13 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_13 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 1. --seed 4
;; 14 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_14 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 1. --seed 5
;; 15 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_15 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 2. --seed 1
;; 16 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_16 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 2. --seed 2
;; 17 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_17 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 2. --seed 3
;; 18 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_18 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 2. --seed 4
;; 19 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_19 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 2. --seed 5
;; 20 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_20 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 5 --seed 1
;; 21 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_21 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 5 --seed 2
;; 22 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_22 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 5 --seed 3
;; 23 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_23 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 5 --seed 4
;; 24 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_24 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 5 --seed 5
;; 25 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_25 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 10 --seed 1
;; 26 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_26 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 10 --seed 2
;; 27 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_27 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 10 --seed 3
;; 28 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_28 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 10 --seed 4
;; 29 )
python3 dcgan_baseline_backpack.py --model resnet --data celeba --local_data --pretrain_exp sep28_dcgan_resnet_bp_imagenet32/run_1 --exp_name oct14_dcgan_celeba_prelim/run_29 --batch_size 256 --lr_gen 3e-4 --lr_dis 3e-4 --n_epochs 10 -gen_freq 10 --clip_norm 1e-4 --target_eps 10 --seed 5
;; esac

CODE=$?
if [ $CODE -eq 0 ]
then
curl -H ta:partying_face -d "oct14_dcgan_celeba_prelim run $1 done." ntfy.sh/frederik_is_cluster_alerts_yeah
elif [ $CODE -ne 3 ]
then
curl -H ta:lady_beetle -d "oct14_dcgan_celeba_prelim run $1 failed. code:$CODE." ntfy.sh/frederik_is_cluster_alerts_ohno
fi
exit $CODE
