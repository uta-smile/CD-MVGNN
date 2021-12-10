CUDA_VISIBLE_DEVICES=3 python train.py --data_path <your data path>/qm7.csv \
    --features_path <your data path>/qm7.npz --no_features_scaling --split_type scaffold_balanced \
    --dataset_type regression --num_folds 10 --ensemble_size 1 --epochs 100 --metric mae --model_type dualmpnnplus \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 --batch_size 32 --depth 5 --hidden_size 2 --ffn_num_layers 4 --ffn_hidden_size 3 \
    --dropout 0.1 --weight_decay 1e-6 --bond_drop_rate 0 --save_dir <your saved dir>