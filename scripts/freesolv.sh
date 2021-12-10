CUDA_VISIBLE_DEVICES=1 python train.py --data_path <your data path>/freesolv.csv \
    --features_path <your data path>/freesolv.npz --no_features_scaling --split_type scaffold_balanced \
    --dataset_type regression --num_folds 10 --ensemble_size 1 --epochs 100 --metric rmse --model_type dualmpnnplus \
    --init_lr 1.5e-4 --max_lr 1e-3 --final_lr 1e-4 --batch_size 32 --depth 5 --hidden_size 4 --ffn_num_layers 4 --ffn_hidden_size 4 \
    --dropout 0.1 --weight_decay 1e-8 --bond_drop_rate 0 --save_dir <your saved dir>