#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH -c 2
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=slurm-CLIP-%x-%j.out

# Activating the conda environment
module load miniforge/24.3.0-0
source activate /nfs/erelab001/shared/Computational_Group/Naravich/OpenAI-CLIP/.venv
which python

python -u convert_clip_back_to_nnUNet.py --config configs/clip_shared_projector_shared_encoder.yaml