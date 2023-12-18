#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=28GB
#SBATCH --gres=gpu
#SBATCH --job-name=torch-single-1

module purge

singularity exec --nv \
	    --overlay /scratch/ssm10076/pytorch-example/cuda_env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; python torch-test.py"
