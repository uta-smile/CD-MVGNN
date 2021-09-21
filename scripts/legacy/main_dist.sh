#!/bin/bash

export LD_LIBRARY_PATH=/conda/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=`pwd`
export PYTHON=/root/anaconda3/bin/python

DATA_ROOT_DIR=/opt/ml/public/readonly/dataset/drug_data
DATA_NAME=tox21
cp ${DATA_ROOT_DIR}/${DATA_NAME}.csv .
cp ${DATA_ROOT_DIR}/${DATA_NAME}.npz .

function train_single_gpu(){
    $PYTHON train.py  --data_path /opt/ml/env/${DATA_NAME}.csv --features_path /opt/ml/env/${DATA_NAME}.npz \
                     --dataset_type classification --save_dir /opt/ml/model/checkpoints \
                     --num_folds 5 --split_type scaffold_balanced --ensemble_size 5 \
                     --no_features_scaling  --bias --dropout 0.3 --ffn_num_layers 6 \
                     --epochs 60 --init_lr 0.0001 --max_lr 10 --final_lr 1 \
                     --batch_size 32 --depth 6 --hidden_size 11 --weight_decay 0.00000001
                     ##ffn_num_layers## --bond_drop_rate ##bond_drop_rate##  --aug_rate ##aug_rate## &
                     ##dropout## --weight_decay ##weight_decay## \
}



function train_multiple_gpus(){
    mpirun --allow-run-as-root -bind-to none \
        -np ${GPU_NUM} \
        -map-by slot \
        -x NCCL_DEBUG=INFO \
        -x NCCL_SOCKET_IFNAME=eth1 \
        -x NCCL_IB_DISABLE=1 \
        $PYTHON train.py  --data_path /opt/ml/env/${DATA_NAME}.csv --features_path /opt/ml/env/${DATA_NAME}.npz \
                         --dataset_type classification --save_dir /opt/ml/model/checkpoints \
                         --num_folds 5 --split_type scaffold_balanced --ensemble_size 5 \
                         --no_features_scaling  --bias --dropout 0.3 --ffn_num_layers 6 \
                         --epochs ${EPOCH_NUM} --init_lr 0.0001 --max_lr 10 --final_lr 1 \
                         --batch_size ${BATCH_SIZE} --depth 6 --hidden_size 11 --weight_decay 0.00000001 \
                         --enbl_multi_gpu
}


GPU_NUM=2
EPOCH_NUM=60
BATCH_SIZE=32
train_multiple_gpus


