#!/bin/bash

#SBATCH -A danielk80_gpu
#SBATCH --partition ica100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name="CS 601.471/671 final project"

module load anaconda
export TRANSFORMERS_CACHE=/scratch4/danielk/schaud31

#init virtual environment if needed
#conda remove -n ppi_pred --all
#conda create -n ppi_pred python=3.9

conda activate ppi_pred # open the Python environment

#conda config --set allow_conda_downgrades true

#clear cache
#pip cache purge

#conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
#pip install -r requirements.txt
#pip install pandas==1.4.4

#conda config --set allow_conda_downgrades false
#conda list

#runs your code
srun python main.py train --batch-size 32 --epochs 10 --lr 1e-4 --small_subset False
