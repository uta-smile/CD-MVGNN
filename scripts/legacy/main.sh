#!/bin/bash

export LD_LIBRARY_PATH=/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/ml/env

dataset_name=##dataset_name##
dataset_csv=$dataset_name.csv
dataset_npz=$dataset_name.npz
cp /opt/ml/disk/drug_data/$dataset_csv .
cp /opt/ml/disk/$dataset_npz .

python train.py  --data_path /opt/ml/env/$dataset_csv --features_path /opt/ml/env/$dataset_npz --no_features_scaling\
                 --dataset_type regression --save_dir /opt/ml/model/checkpoints \
                 --num_folds 5 --split_type random --ensemble_size 5  --model_type mpnn\
                 --epochs 40 --init_lr ##init_lr## --max_lr 10 --final_lr 1 \
                 --batch_size 32  --depth ##depth## --hidden_size ##hidden_size## \
                 --dropout ##dropout## --weight_decay ##weight_decay## \
                 --ffn_num_layers ##ffn_num_layers## --ffn_hidden_size ##ffn_hidden_size## &

python turbo.py &

wait -n
pkill -P $$


