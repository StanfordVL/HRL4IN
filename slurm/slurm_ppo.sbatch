#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#partition name
#SBATCH --partition=napoli-gpu
#################
#number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#################  
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=/cvgl2/u/yveys/Logs/%j.out
#################
# a file for errors from the job
#SBATCH --error=/cvgl2/u/yveys/Logs/%j.err
#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm
#SBATCH --time=12:00:00
#################
# Quality of Service (QOS); think of it as sending your job into a special queue; --qos=long for with a max job length of 7 days.
# uncomment ##SBATCH --qos=long if you want your job to run longer than 48 hours, which is the default for normal partition,  
# NOTE- in the hns partition the default max run time is 7 days , so you wont need to include qos, also change to normal partition 
# since dev max run time is 2 hours.
##SBATCH --qos=long
# We are submitting to the dev partition, there are several on sherlock: normal, gpu, bigmem (jobs requiring >64Gigs RAM) 
##SBATCH -p dev 
#################
# --mem is memory per node; default is 4000 MB per CPU, remember to ask for enough mem to match your CPU request, since 
# sherlock automatically allocates 4 Gigs of RAM/CPU, if you ask for 8 CPUs you will get 32 Gigs of RAM, so either 
# leave --mem commented out or request >= to the RAM needed for your CPU request.  It will also accept mem. in units, ie "--mem=4G"
#SBATCH --mem=12000
# to request multiple threads/CPUs use the -c option, on Sherlock we use 1 thread/CPU, 16 CPUs on each normal compute node 4Gigs RAM per CPU.  Here we will request just 1.
#SBATCH -c 1
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
#SBATCH --mail-type=BEGIN,END,FAIL # notifications for job done & fail
# Remember to change this to your email
#SBATCH --mail-user=yveys@caltech.edu
# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
#now run normal batch commands
# Begins in slurm_scripts
source ~/.bashrc
cd /cvgl2/u/yveys/anaconda3/bin/
source activate ../envs/py3-igibson
export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard'

cd /cvgl2/u/yveys/iGibson
git pull

cd /cvgl2/u/yveys/MMLearning
#git pull

cd /cvgl2/u/yveys/MMLearning/hrl4in
./run_train_ppo.sh

# done
echo "Done"
exit 0
