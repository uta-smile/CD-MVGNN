#!/bin/bash
dataset=##dataset##
PYTHON=/root/anaconda3/bin/python
data_path=/opt/ml/disk/drug_data/${dataset}.csv
features_path=/opt/ml/disk/drug_data/${dataset}.npz
save_dir=/opt/ml/env/model_dir/${dataset}
mkdir -p ${save_dir}

init_lr=##init_lr##
max_lr=$(python -c "print(##init_lr##*##max_lr##, end='')")
final_lr=$(python -c "print(##init_lr##*##final_lr##, end='')")
echo ${init_lr}
echo ${max_lr}
echo ${final_lr}

$PYTHON train.py \
    --data_path ${data_path} \
    --features_path ${features_path} \
    --dataset_type ##dataset_type## \
    --save_dir ${save_dir} ##checkpoint_path## \
    --fine_tune_coff ##fine_tune_coff## \
    --no_features_scaling \
    --num_folds 5 \
    --split_type ##split_type## \
    --metric ##metric## \
    --ensemble_size 1 \
    --model_type ##model_type## \
    --batch_size 32 \
    --dropout ##dropout## \
    --activation PReLU \
    --depth ##depth## \
    --ffn_hidden_size ##ffn_hidden_size## \
    --ffn_num_layers ##ffn_num_layers## \
    --hidden_size ##hidden_size## \
    --epochs 140 \
    --init_lr ${init_lr} \
    --max_lr ${max_lr} \
    --final_lr ${final_lr} \
    --bond_drop_rate ##bond_drop_rate## \
    --weight_decay ##weight_decay## \
    --self_attention  --attn_hidden ##attn_hidden## \
    --attn_out ##attn_out## \
    --aug_rate ##aug_rate## \
    --gpu 0 \
    ##select_by_loss##  ##no_attach_fea## \
