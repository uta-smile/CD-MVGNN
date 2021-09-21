#!/bin/bash
GPU_NUM=N_GPU
PYTHON=python
data_path=/root/data/drug_data/grover_data/molbertpretrain.csv
fg_label_path=/root/data/drug_data/grover_data/molbertpretrain_molbert.npz
vocab_path=/root/data/drug_data/grover_data/all_smi_vocab.pkl
save_dir=/root/data/model_dmpnn/checkpoints/molbert

epochs=20
warmup_epochs=2
batch_size=512

init_lr=0.0002
max_lr=10
final_lr=2
hidden_size=5
depth=5
dropout=0
activation=ReLU
weight_decay=0.0000001
select_by_loss=--select_by_loss
no_attach_fea=--no_attach_fea

export OMPI_MCA_btl_vader_single_copy_mechanism=none

python -V

CUDA_VISIBLE_DEVICES=1,5 mpirun --allow-run-as-root -bind-to none \
    -np ${GPU_NUM} \
    -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=eth1 \
    -x NCCL_IB_DISABLE=1 \
    -x PYTHONPATH=`pwd` \
    $PYTHON dglt/contrib/grover/run.py --enable_multi_gpu --data_path ${data_path} --fg_label_path ${fg_label_path} --vocab_path ${vocab_path} \
                     --save_dir ${save_dir} --split_type random  \
                     --epochs ${epochs} --warmup_epochs ${warmup_epochs} \
                     --batch_size ${batch_size} \
                     --init_lr ${init_lr} --max_lr ${max_lr} --final_lr ${final_lr} --dropout ${dropout} \
                     --depth ${depth} --hidden_size ${hidden_size} --weight_decay ${weight_decay} ${select_by_loss} ${no_attach_fea} \
