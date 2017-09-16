#!/bin/sh
export cpu_or_gpu="gpu"
export device_num=0
sbatch --partition=gpu -w scf-sm20-gpu --gres=gpu:1 /accounts/projects/vision/chandan/v4_natural_movies/scripts/extract.sh
