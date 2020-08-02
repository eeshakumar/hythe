#!/bin/bash
#SBATCH --qos normal_students
#SBATCH -c 4
##SBATCH --gres=gpu:1
#SBATCH --mem=10GB

srun singularity exec --nv ../images/hythe_latest.sif python3 -u ./$1