#!/bin/bash

gpu="0"
reward_type="dense"
pos="random"
lr="1e-4"
num_steps="1024"
run="jr_interactive_nav"

log_dir="reward_"$reward_type"_pos_"$pos"_lr_"$lr"_nsteps_"$num_steps"_run_"$run
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
   --checkpoint-index 870 \
   --experiment-folder "ckpt/"$log_dir \
   --checkpoint-interval 10 \
   --env-type "gibson" \
   --config-file "jr_interactive_nav.yaml" \
   --arena "stadium" \
   --num-eval-episodes 100 \
   --env-mode "gui" \
   --eval-only \
   --random-height

