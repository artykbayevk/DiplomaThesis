#!/bin/bash -l

#SBATCH
#SBATCH --job-name=vgg19
#SBATCH --time=3-00:00:00
#SBATCH --partition=dpart
#SBATCH --nodelist=vulcan02
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:p6000:1
#SBATCH --mail-type=end
#SBATCH --mail-user=kamalkhan.sdu@gmail.com


#module add cuda/8.0.44 cudnn/v5.1

python fl47main.py

#/fs/vulcan-scratch/kamalsdu/caffe/build/tools/caffe train -solver solver.prototxt

echo $SLURM_JOBID
echo $SLURN_SUBMIT_DIRECTORY
echo $SLURM_SUBMIT_HOST
echo $SLURM_JOB_NODELIST
echo "Finished with job $SLURM_JOBID"
