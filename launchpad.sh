#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="CS 601.471/671 final project"

module load anaconda
export TRANSFORMERS_CACHE=/scratch4/danielk/schaud31

#init virtual environment if needed
#conda remove -n ppi_pred --all
conda create -n ppi_pred python=3.9

conda activate ppi_pred # open the Python environment

conda config --set allow_conda_downgrades true

#clear cache
pip cache purge

pip install torch torchvision torchaudio
pip install -r requirements.txt
conda config --set allow_conda_downgrades false
#conda list

#clear cache
#pip install huggingface_hub["cli"]
#huggingface-cli delete-cache

#runs your code
srun python main.py train --batch-size 32 --epochs 10 --lr 1e-4 --small_subset True
