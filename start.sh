#!/bin/bash
export PATH="/root/anaconda3/bin:$PATH"
export LD_LIBRARY_PATH="/root/anaconda3/lib:$LD_LIBRARY_PATH"



python train.py --config_path job_param.json \
                --no_features_scaling  \
                --batch_size 32  --gpu 0 --weight_decay 1e-10
