#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 10
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=slurm-CLIP-%x-%j.out

# Activating the conda environment
module load miniforge/24.3.0-0
source activate /nfs/erelab001/shared/Computational_Group/Naravich/OpenAI-CLIP/.venv
which python

python main.py --config configs/configs/clip_shared_projector_shared_encoder_preivl_poststent_wd.yaml