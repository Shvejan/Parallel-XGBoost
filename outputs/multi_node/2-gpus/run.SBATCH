#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=28GB
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --job-name=b2

module purge

singularity exec --nv \
	    --overlay /scratch/ssm10076/pytorch-example/cuda_env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; python /scratch/ssm10076/pytorch-example/Parallel-XGBoost/outputs/multi_node/2-gpus/train-code.py"
