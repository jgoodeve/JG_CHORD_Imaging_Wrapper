#!/bin/bash 
#SBATCH --account=def-acliu
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem=0M                # memory (per node)
#SBATCH --time=0-01:00          # time (DD-HH:MM)

module --force purge
module use /project/rrg-kmsmith/shared/chord_env/modules/modulefiles/
module load chord/chord_pipeline/2023.06
module load cudacore/.12.2.2

cd ~/scratch/jgoodeve/cuda_dirtymap_josh
python pyscript_template.py
