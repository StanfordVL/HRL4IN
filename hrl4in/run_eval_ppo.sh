#!/bin/bash

#gpu="0"
#pos="rh"
#reward_type="l2"
#tol=0.1
#success_reward=100.0
#potential_reward=1.0
#gamma=0.99 
#lr="1e-4"
#num_steps="500"
#speed="0.075_0.4"

#log_dir="pos_"$pos"_tol_"$tol"_suc_rwd_"$success_reward"_pot_rwd_"$potential_reward"_gma_"$gamma"_lr_"$lr"_nstps_"$num_steps"_spd_"$speed
#echo $log_dir

gpu="0"
pos="fix_s14_1.2"
reward_type="l2"
tol=0.2
success_reward=10.0
potential_reward=30.0
col_reward=0.0
gamma=0.99 
lr="1e-4"
num_steps="60"
speed="0.25_0.25"

log_dir="pos_"$pos"_tol_"$tol"_suc_rwd_"$success_reward"_pot_rwd_"$potential_reward"_col_rwd_"$col_reward"_gma_"$gamma"_lr_"$lr"_nstps_"$num_steps"_spd_"$speed
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
   --num-updates 100000 \
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
   --num-eval-episodes 100 \
   --env-mode "pbgui" \
   --eval-only \
   --gamma $gamma \
   --random-height
