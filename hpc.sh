#!/bin/bash
#
#SBATCH --job-name=pt-transformer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#

cd /scratch/hw2588/pt-transformer
python main.py