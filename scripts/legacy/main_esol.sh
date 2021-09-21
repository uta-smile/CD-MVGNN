#!/bin/bash
python train.py --data_path ./datasets/all.csv \
                --features_path ./datasets/all.npz \
                --no_features_scaling \
                --dataset_type regression --epochs 300 --batch_size 32 \
                --dropout 0.15 --ffn_num_layers 2 --hidden_size 11 --ffn_hidden_size 700 \
                --depth 6 --num_folds 1 --ensemble_size 1 --split_type random \
                --save_dir ./model

cp ./model/fold_0/model_0/model.pt .

python train.py  --data_path ./datasets/delaney.csv \
                 --features_path ./datasets/delaney.npz \
                 --checkpoint_path ./model.pt  --no_features_scaling \
                 --dataset_type regression --epochs 30 --batch_size 32 \
                 --dropout 0.15 --ffn_num_layers 2 --hidden_size 11 --ffn_hidden_size 700 \
                 --depth 6 --num_folds 5 --ensemble_size 1 --split_type random \
                 --save_dir ./model --init_lr 1e-5

