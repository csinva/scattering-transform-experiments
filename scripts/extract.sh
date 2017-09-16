#!/bin/sh -l
module load tensorflow
python /accounts/projects/vision/chandan/v4_natural_movies/extract.py --device "/$cpu_or_gpu:$device_num" 
