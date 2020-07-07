#!/bin/bash

gpu="0"
pos="rand_h"
reward_type="l2"
tol=0.06
success_reward=10.0
potential_reward=1.0
gamma=0.99 
lr="1e-4"
num_steps="250"

log_dir="pos_"$pos"_reward_"$reward_type"_tol_"$tol"_suc_rew_"$success_reward"_pot_rew_"$potential_reward"_gamma_"$gamma"_lr_"$lr"_nsteps_"$num_steps
echo $log_dir

python -u train_ppo.py \
   --use-gae \
   --sim-gpu-id $gpu \
   --pth-gpu-id $gpu \
   --lr $lr \
   --clip-param 0.1 \
   --value-loss-coef 0.5 \
   --num-train-processes 1 \
   --num-eval-processes 1 \
   --num-steps $num_steps \
   --num-mini-batch 1 \
   --num-updates 50000 \
   --use-linear-lr-decay \
   --use-linear-clip-decay \
   --entropy-coef 0.01 \
   --log-interval 1 \
   --experiment-folder "ckpt/"$log_dir \
   --checkpoint-interval 10 \
   --checkpoint-index -1 \
   --env-type "gibson" \
   --config-file $log_dir".yaml" \
   --arena "stadium" \
   --num-eval-episodes 1 \
   --env-mode "gui" \
   --eval-only \
   --gamma $gamma \
   --random-height
