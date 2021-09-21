#!/bin/bash

export LD_LIBRARY_PATH=/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/ml/env

dataset_name=muv
dataset_csv=$dataset_name.csv
dataset_npz=$dataset_name.npz
cp /opt/ml/disk/drug_data/$dataset_csv .
cp /opt/ml/disk/drug_data/$dataset_npz .

python train.py  --data_path /opt/ml/env/$dataset_csv --features_path /opt/ml/env/$dataset_npz \
                 --dataset_type classification --save_dir /opt/ml/model/ \
                 --num_folds 1 --split_type scaffold_balanced --ensemble_size 1 --batch_size 32 \
                 --no_features_scaling \
                 --dropout 0 --depth 3 --ffn_num_layers 2 --hidden_size 3 \
                 --epochs 30  --init_lr 0.0001 --max_lr 10 --final_lr 1 --metric prc-auc --no_cuda 
