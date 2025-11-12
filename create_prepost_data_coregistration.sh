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

python create_prepost_data_coregistration.py