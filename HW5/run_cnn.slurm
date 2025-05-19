#!/bin/bash
#SBATCH --job-name=cnn_hw5
#SBATCH --partition=gpu1
#SBATCH --output=cnn_hw5.out
#SBATCH --error=cnn_hw5.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module load python39
source ml_hw5_env/bin/activate

python cnn_crack_detection_fast.py

