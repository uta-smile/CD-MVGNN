#!/bin/bash

export LD_LIBRARY_PATH=/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/ml/env

dataset_name=sider
dataset_csv=$dataset_name.csv
dataset_pkl=$dataset_name.pkl
#cp /opt/ml/disk/drug_data/$dataset_csv .
#cp /opt/ml/disk/drug_data/$dataset_pkl .

CUDA_VISIBLE_DEVICES=1 python train.py  --data_path data/$dataset_csv --dataset_type classification --save_dir model/checkpoints \
                 --num_folds 5  --split_type scaffold_balanced --ensemble_size 5 --batch_size 8 \
                 --features_generator rdkit_2d_normalized --no_features_scaling \
                 --dropout 0.5 --depth 6 --ffn_num_layers 1 --hidden_size 8 \
                 --epochs 100  --init_lr 0.00015 --max_lr 10 --final_lr 2 #--bias
#sleep 5

#python train.py  --data_path /opt/ml/env/$dataset_csv --dataset_type classification --save_dir /opt/ml/model/checkpointr \
#                 --num_folds 5 --split_type random --ensemble_size 5 --batch_size 32 \
#                 --features_generator rdkit_2d_normalized --no_features_scaling \
#                 --dropout 0.05 --depth 5 --ffn_num_layers 1 --hidden_size 14 \
#                 --epochs 30  --init_lr 0.0001 --max_lr 10 --final_lr 1  --no_cache &

#wait
