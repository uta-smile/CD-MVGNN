#!/bin/bash

export LD_LIBRARY_PATH=/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/ml/env

dataset_name=##dataset_name##
dataset_csv=$dataset_name.csv
dataset_npz=$dataset_name.npz
cp /opt/ml/disk/drug_data/$dataset_csv .
cp /opt/ml/disk/drug_data/$dataset_npz .

python train.py  --data_path /opt/ml/env/$dataset_csv --features_path /opt/ml/env/$dataset_npz --no_features_scaling\
                 --dataset_type classification --save_dir /opt/ml/model/checkpoints \
                 --num_folds 5 --split_type scaffold_balanced --ensemble_size 5 --select_by_loss --bias \
                 --epochs 100 --init_lr ##init_lr## --max_lr 10 --final_lr 1 \
                 --batch_size 32  --depth ##depth## --hidden_size ##hidden_size## \
                 --dropout ##dropout## --weight_decay ##weight_decay## \
                 --ffn_num_layers ##ffn_num_layers## --ffn_hidden_size ##ffn_hidden_size## \
                 --bond_drop_rate ##bond_drop_rate## --aug_rate ##aug_rate## \
                 --self_attention --attn_hidden ##attn_hidden## --attn_out ##attn_out## \
                 --dist_coff ##dist_coff## ##no_attach_fea## &

python turbo.py &

wait -n
pkill -P $$


