CUDA_VISIBLE_DEVICES=0 python train.py --data_path <your data path>/lipo.csv \
    --features_path <your data path>/lipo.npz --no_features_scaling --split_type scaffold_balanced \
    --dataset_type regression --num_folds 10 --ensemble_size 1 --epochs 100 --metric rmse --model_type dualmpnnplus \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1.5e-4 --batch_size 32 --depth 3 --hidden_size 3 --ffn_num_layers 2 --ffn_hidden_size 1 \
    --dropout 0 --weight_decay 1e-7 --bond_drop_rate 0 --no_attach_fea --save_dir <your saved dir>