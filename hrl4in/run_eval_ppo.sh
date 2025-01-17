#!/bin/bash

gpu="0"
pos="fixed"
reward_type="l2"
tol=0.5
success_reward=30.0
potential_reward=5.0
col_reward=-5.0
gamma=0.99
lr="1e-4"
num_steps="200"
speed="0.2_0"

log_dir="RUN_reaching_2_obs_split_net_seed_1"
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
   --num-updates 1000000 \
   --use-linear-lr-decay \
   --use-linear-clip-decay \
   --entropy-coef 0.01 \
   --log-interval 1 \
   --experiment-folder "ckpt/"$log_dir \
   --checkpoint-interval 200 \
   --checkpoint-index -1 \
   --env-type "gibson" \
   --config-file "master_config.yaml" \
   --arena "stadium" \
   --num-eval-episodes 100 \
   --env-mode "gui" \
   --eval-only \
   --gamma $gamma \
   --random-pos \
   --use-camera-masks \
   --split-network
