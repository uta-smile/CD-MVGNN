#!/bin/bash

#export LD_LIBRARY_PATH=/conda/lib:$LD_LIBRARY_PATH
#export PYTHONPATH=/opt/ml/env

DATA_ROOT_DIR=../mit/drug_data
DATA_NAME=tox21
cp ${DATA_ROOT_DIR}/${DATA_NAME}.csv .
cp ${DATA_ROOT_DIR}/${DATA_NAME}.npz .
OUT_DIR=/out
mkdir -p ${OUT_DIR}

export CUDA_VISIBLE_DEVICES='1, 0'

python train.py  --data_path ${DATA_ROOT_DIR}/${DATA_NAME}.csv --features_path ${DATA_ROOT_DIR}/${DATA_NAME}.npz \
                 --dataset_type classification --save_dir ${OUT_DIR}/checkpoints \
                 --num_folds 5 --split_type scaffold_balanced --ensemble_size 5 \
                 --no_features_scaling  --bias --dropout 0.3 --ffn_num_layers 6 \
                 --epochs 60 --init_lr 0.0001 --max_lr 10 --final_lr 1 \
                 --batch_size 128  --depth 6 --hidden_size 11 --weight_decay 0.00000001 \
                 --enbl_multi_gpu
                 ##ffn_num_layers## --bond_drop_rate ##bond_drop_rate##  --aug_rate ##aug_rate## &
                 ##dropout## --weight_decay ##weight_decay## \

