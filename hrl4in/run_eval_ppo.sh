#!/bin/bash

gpu="0"
pos="wall_fov_90_mov_short"
reward_type="l2"
tol=0.1
success_reward=30.0
potential_reward=5.0
col_reward=-10.0
gamma=0.99 
lr="1e-4"
num_steps="250"
speed="0.1_0.1"

log_dir="jr2_"$pos"_tol_"$tol"_suc_rwd_"$success_reward"_pot_rwd_"$potential_reward"_col_rwd_"$col_reward"_gma_"$gamma"_lr_"$lr"_nstps_"$num_steps"_spd_"$speed
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
   --config-file $log_dir".yaml" \
   --arena "stadium" \
   --num-eval-episodes 100 \
   --env-mode "gui" \
   --eval-only \
   --gamma $gamma \
   --random-pos
