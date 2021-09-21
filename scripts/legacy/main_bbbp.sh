#!/bin/bash

dataset_name=bbbp
dataset_csv=$dataset_name.csv
dataset_npz=$dataset_name.npz

#python train.py  --data_path /data2/weiyangxie/mit/drug_data/bbbp.csv --features_path /data2/weiyangxie/mit/drug_data/bbbp.npz \
#                 --dataset_type classification --save_dir ./checkpoints \
#                 --num_folds 5 --split_type scaffold_balanced --ensemble_size 5 \
#                 --no_features_scaling  --bias  --weight_decay 0.00000001 \
#                 --epochs 60 --init_lr 0.0001 --max_lr 10 --final_lr 1 --batch_size 32 \
#                 --dropout 0.242  --depth 6 --hidden_size 5 \
#                 --ffn_num_layers 6 --gpu 2
python train.py --data_path /data2/weiyangxie/mit/drug_data/bbbp.csv --features_path /data2/weiyangxie/mit/drug_data/bbbp.npz --no_features_scaling --epochs 30 --batch_size 32 --dropout 0.3 --ffn_num_layers 6 --hidden_size 11 --depth 6 --num_folds 5 --ensemble_size 5 --split_type scaffold_balanced --gpu 3 --save_dir ./model 
#
