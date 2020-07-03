#!/bin/bash

gpu="0"
reward_type="dense"
pos="fixed"
irs="30.0"
sgr="0.0"
lr="1e-4"
meta_lr="1e-5"        # 1e-4, 1e-5
fr_lr="0"             # 0, 100
death="30.0"
init_std_dev_xy="0.6" # 0.6, 1.2
init_std_dev_z="0.1"
failed_pnt="0.0"      # 0.0, -0.2
num_steps="1024"
ext_col="0.0"         # 0.0, 0.5, 1.0, 2.0
name="exp"
run="4"

log_dir="hrl_reward_"$reward_type"_pos_"$pos"_sgm_arm_world_irs_"$irs"_sgr_"$sgr"_lr_"$lr"_meta_lr_"$meta_lr"_fr_lr_"$fr_lr"_death_"$death"_init_std_"$init_std_dev_xy"_"$init_std_dev_xy"_"$init_std_dev_z"_failed_pnt_"$failed_pnt"_nsteps_"$num_steps"_ext_col_"$ext_col"_6x6_from_scr_"$name"_run_"$run
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
   --config-file "jr2_reaching.yaml" \
   --arena "stadium" \
   --num-eval-episodes 1 \
   --env-mode "headless" \
   --random-height
