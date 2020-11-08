#!/bin/bash
#SBATCH --qos lowprio_students
#SBATCH -c 4
##SBATCH --gres=gpu:1
#SBATCH --mem=10GB

srun singularity exec --nv /mnt/glusterdata/home/ekumar/images/hythe_latest_2.sif python3 -u ./$1 $2