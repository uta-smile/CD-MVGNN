#!/bin/bash 
echo $N_GPU
GPU_NUM=$N_GPU
PYTHON=/root/anaconda3/bin/python
data_path=/opt/ml/disk/grover_data/all_smi_filtered.csv
fg_label_path=/opt/ml/disk/grover_data/all_smi_filtered_molbert.npz
vocab_path=/opt/ml/disk/grover_data/all_smi_vocab.pkl
#save_dir=/opt/ml/env/model_dir/grover/

#data_path=/opt/ml/disk/grover_data/molbertpretrain.csv
#fg_label_path=/opt/ml/disk/grover_data/molbertpretrain_molbert.npz
#vocab_path=/opt/ml/disk/grover_data/all_smi_vocab.pkl
save_dir=/opt/ml/disk/grover/median/
mkdir -p save_dir

epochs=500
warmup_epochs=2
batch_size=256

init_lr=0.00015
max_lr=5
final_lr=1
hidden_size=19
depth=6
dropout=0
activation=ReLU
weight_decay=0.0000001
select_by_loss=--select_by_loss
no_attach_fea=--no_attach_fea

export OMPI_MCA_btl_vader_single_copy_mechanism=none

#-np ${GPU_NUM} \

mpirun --allow-run-as-root -bind-to none \
    -map-by slot \
    -mca btl_tcp_if_include eth1 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=eth1 \
    -x NCCL_IB_DISABLE=1 \
    -x PYTHONPATH=`pwd` \
    $PYTHON dglt/contrib/grover/run.py --enable_multi_gpu --data_path ${data_path} --fg_label_path ${fg_label_path} --vocab_path ${vocab_path} \
                     --save_dir ${save_dir} --split_type random  --save_interval 100 \
                     --epochs ${epochs} --warmup_epochs ${warmup_epochs} \
                     --batch_size ${batch_size} \
                     --init_lr ${init_lr} --max_lr ${max_lr} --final_lr ${final_lr} \
                     --dropout ${dropout} --activation ${activation} \
                     --depth ${depth} --hidden_size ${hidden_size} --weight_decay ${weight_decay} ${select_by_loss} ${no_attach_fea} \
