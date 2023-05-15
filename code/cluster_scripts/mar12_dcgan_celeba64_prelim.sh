cp -r ../data/img_align_celeba /tmp/img_align_celeba
#!/bin/bash
case $1 in
0 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_0 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 10. --seed 1
;; 1 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_1 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 10. --seed 2
;; 2 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_2 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 10. --seed 3
;; 3 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_3 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 5. --seed 1
;; 4 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_4 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 5. --seed 2
;; 5 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_5 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 5. --seed 3
;; 6 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_6 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 2. --seed 1
;; 7 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_7 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 2. --seed 2
;; 8 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_8 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 2. --seed 3
;; 9 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_9 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 1. --seed 1
;; 10 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_10 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 1. --seed 2
;; 11 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_11 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 1. --seed 3
;; 12 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_12 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 0.5 --seed 1
;; 13 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_13 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 0.5 --seed 2
;; 14 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_14 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 0.5 --seed 3
;; 15 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_15 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 0.2 --seed 1
;; 16 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_16 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 0.2 --seed 2
;; 17 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_17 --pretrain_exp mar8_dcgan_imagenet64/run_130 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 0.2 --seed 3
;; 18 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_18 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 10. --seed 1
;; 19 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_19 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 10. --seed 2
;; 20 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_20 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 10. --seed 3
;; 21 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_21 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 5. --seed 1
;; 22 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_22 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 5. --seed 2
;; 23 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_23 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 5. --seed 3
;; 24 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_24 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 2. --seed 1
;; 25 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_25 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 2. --seed 2
;; 26 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_26 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 2. --seed 3
;; 27 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_27 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 1. --seed 1
;; 28 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_28 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 1. --seed 2
;; 29 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_29 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 1. --seed 3
;; 30 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_30 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 0.5 --seed 1
;; 31 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_31 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 0.5 --seed 2
;; 32 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_32 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 0.5 --seed 3
;; 33 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_33 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 0.2 --seed 1
;; 34 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_34 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 0.2 --seed 2
;; 35 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --batch_size_grad_acc 64 --exp_name mar12_dcgan_celeba64_prelim/run_35 --pretrain_exp mar8_dcgan_imagenet64/run_61 --batch_size 256 --lr_gen 3e-5 --lr_dis 3e-3 --n_epochs 10 -gen_freq 1 --clip_norm 1e-5 --target_eps 0.2 --seed 3
;; esac

CODE=$?
if [ $CODE -eq 0 ]
then
curl -H ta:partying_face -d "mar12_dcgan_celeba64_prelim run $1 done." ntfy.sh/frederik_is_cluster_alerts_yeah
elif [ $CODE -ne 3 ]
then
curl -H ta:lady_beetle -d "mar12_dcgan_celeba64_prelim run $1 failed. code:$CODE." ntfy.sh/frederik_is_cluster_alerts_ohno
fi
exit $CODE
