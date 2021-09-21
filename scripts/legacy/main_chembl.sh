#!/bin/bash

export LD_LIBRARY_PATH=/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/ml/env

dataset_name=chembl
dataset_csv=$dataset_name.csv
dataset_pkl=$dataset_name.pkl
cp /opt/ml/disk/drug_data/$dataset_csv .
cp /opt/ml/disk/drug_data/$dataset_pkl .

python train.py  --data_path /opt/ml/env/$dataset_csv --dataset_type classification --save_dir checkpoints \
                 --num_folds 5 --split_type scaffold_balanced --ensemble_size 5 --batch_size 320 \
                 --features_generator rdkit_2d_normalized --no_features_scaling \
                 --dropout 0.05 --depth 6 --ffn_num_layers 2 --hidden_size 7 \
                 --epochs 30  --init_lr 0.0001 --max_lr 10 --final_lr 1  --no_cache 
#sleep 5

#python train.py  --data_path /opt/ml/env/$dataset_csv --dataset_type classification --save_dir checkpointr \
#                 --num_folds 5 --split_type random --ensemble_size 5 --batch_size 320 \
#                 --features_generator rdkit_2d_normalized --no_features_scaling \
#                 --dropout 0.05 --depth 6 --ffn_num_layers 2 --hidden_size 7 \
#                 --epochs 30  --init_lr 0.0001 --max_lr 10 --final_lr 1  --no_cache




