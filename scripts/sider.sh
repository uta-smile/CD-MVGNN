CUDA_VISIBLE_DEVICES=2 python train.py --data_path <your data path>/sider.csv \
    --features_path <your data path>/sider.npz --no_features_scaling --split_type scaffold_balanced \
    --dataset_type classification --num_folds 10  --ensemble_size 1 --epochs 100 --metric auc --model_type dualmpnnplus \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 --batch_size 32 --depth 3 --hidden_size 5 --ffn_num_layers 4 --ffn_hidden_size 2 \
    --dropout 0 --weight_decay 1e-6 --bond_drop_rate 0 --no_attach_fea --save_dir <your saved dir>