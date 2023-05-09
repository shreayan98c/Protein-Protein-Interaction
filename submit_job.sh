#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --account=jgray21_gpu
#SBATCH --time=24:00:00
#SBATCH --mem=40GB

python main.py train --batch-size 16 --epochs 10 --lr 1e-4 --small_subset False