#!/bin/bash
#SBATCH --qos lowprio_students
#SBATCH -c 4
##SBATCH --gres=gpu:1
#SBATCH --mem=10GB

num_cpus=25
memory=25

export hy_slurm_num_cpus=$num_cpus
export hy_slurm_memory=$memory
echo "Using cpus = $hy_slurm_num_cpus"
echo "Using memory = $hy_slurm_memory"

srun singularity exec --nv /mnt/glusterdata/home/ekumar/images/hythe_latest_2.sif python3 -u ./$1 $2