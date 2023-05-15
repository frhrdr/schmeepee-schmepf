cp -r ../data/img_align_celeba /tmp/img_align_celeba
#!/bin/bash
case $1 in
0 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --pretrain_exp mar8_dcgan_imagenet64/run_130  --batch_size_grad_acc 64 --exp_name mar13_dcgan_celeba64_eps1_and_05_prelim/run_0 --batch_size 256 --n_epochs 10  --lr_gen 3e-4 --lr_dis 1e-3 -gen_freq 30 --clip_norm 1e-3 --target_eps 0.5 --seed 1
;; 1 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --pretrain_exp mar8_dcgan_imagenet64/run_130  --batch_size_grad_acc 64 --exp_name mar13_dcgan_celeba64_eps1_and_05_prelim/run_1 --batch_size 256 --n_epochs 10  --lr_gen 3e-4 --lr_dis 1e-3 -gen_freq 30 --clip_norm 1e-3 --target_eps 0.5 --seed 2
;; 2 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --pretrain_exp mar8_dcgan_imagenet64/run_130  --batch_size_grad_acc 64 --exp_name mar13_dcgan_celeba64_eps1_and_05_prelim/run_2 --batch_size 256 --n_epochs 10  --lr_gen 3e-4 --lr_dis 1e-3 -gen_freq 30 --clip_norm 1e-3 --target_eps 0.5 --seed 3
;; 3 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --pretrain_exp mar8_dcgan_imagenet64/run_130  --batch_size_grad_acc 64 --exp_name mar13_dcgan_celeba64_eps1_and_05_prelim/run_3 --batch_size 256 --n_epochs 10  --lr_gen 3e-5 --lr_dis 1e-3 -gen_freq 10 --clip_norm 1e-5 --target_eps 0.5 --seed 1
;; 4 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --pretrain_exp mar8_dcgan_imagenet64/run_130  --batch_size_grad_acc 64 --exp_name mar13_dcgan_celeba64_eps1_and_05_prelim/run_4 --batch_size 256 --n_epochs 10  --lr_gen 3e-5 --lr_dis 1e-3 -gen_freq 10 --clip_norm 1e-5 --target_eps 0.5 --seed 2
;; 5 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --pretrain_exp mar8_dcgan_imagenet64/run_130  --batch_size_grad_acc 64 --exp_name mar13_dcgan_celeba64_eps1_and_05_prelim/run_5 --batch_size 256 --n_epochs 10  --lr_gen 3e-5 --lr_dis 1e-3 -gen_freq 10 --clip_norm 1e-5 --target_eps 0.5 --seed 3
;; 6 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --pretrain_exp mar8_dcgan_imagenet64/run_130  --batch_size_grad_acc 64 --exp_name mar13_dcgan_celeba64_eps1_and_05_prelim/run_6 --batch_size 256 --n_epochs 10  --lr_gen 3e-4 --lr_dis 1e-3 -gen_freq 30 --clip_norm 1e-3 --target_eps 1. --seed 1
;; 7 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --pretrain_exp mar8_dcgan_imagenet64/run_130  --batch_size_grad_acc 64 --exp_name mar13_dcgan_celeba64_eps1_and_05_prelim/run_7 --batch_size 256 --n_epochs 10  --lr_gen 3e-4 --lr_dis 1e-3 -gen_freq 30 --clip_norm 1e-3 --target_eps 1. --seed 2
;; 8 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --pretrain_exp mar8_dcgan_imagenet64/run_130  --batch_size_grad_acc 64 --exp_name mar13_dcgan_celeba64_eps1_and_05_prelim/run_8 --batch_size 256 --n_epochs 10  --lr_gen 3e-4 --lr_dis 1e-3 -gen_freq 30 --clip_norm 1e-3 --target_eps 1. --seed 3
;; 9 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --pretrain_exp mar8_dcgan_imagenet64/run_130  --batch_size_grad_acc 64 --exp_name mar13_dcgan_celeba64_eps1_and_05_prelim/run_9 --batch_size 256 --n_epochs 10  --lr_gen 3e-4 --lr_dis 1e-3 -gen_freq 30 --clip_norm 1e-4 --target_eps 1. --seed 1
;; 10 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --pretrain_exp mar8_dcgan_imagenet64/run_130  --batch_size_grad_acc 64 --exp_name mar13_dcgan_celeba64_eps1_and_05_prelim/run_10 --batch_size 256 --n_epochs 10  --lr_gen 3e-4 --lr_dis 1e-3 -gen_freq 30 --clip_norm 1e-4 --target_eps 1. --seed 2
;; 11 )
python3.9 dcgan_baseline_backpack.py --model resnet --data celeba64 --local_data --pretrain_exp mar8_dcgan_imagenet64/run_130  --batch_size_grad_acc 64 --exp_name mar13_dcgan_celeba64_eps1_and_05_prelim/run_11 --batch_size 256 --n_epochs 10  --lr_gen 3e-4 --lr_dis 1e-3 -gen_freq 30 --clip_norm 1e-4 --target_eps 1. --seed 3
;; esac

CODE=$?
if [ $CODE -eq 0 ]
then
curl -H ta:partying_face -d "mar13_dcgan_celeba64_eps1_and_05_prelim run $1 done." ntfy.sh/frederik_is_cluster_alerts_yeah
elif [ $CODE -ne 3 ]
then
curl -H ta:lady_beetle -d "mar13_dcgan_celeba64_eps1_and_05_prelim run $1 failed. code:$CODE." ntfy.sh/frederik_is_cluster_alerts_ohno
fi
exit $CODE
