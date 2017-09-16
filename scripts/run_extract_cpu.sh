#!/bin/sh
export cpu_or_gpu="cpu"
export device_num=0
sbatch --partition=low /accounts/projects/vision/chandan/v4_natural_movies/scripts/extract.sh
