CUDA_VISIBLE_DEVICES=1 python train.py --data_path <your data path>/toxcast.csv \
    --features_path <your data path>/toxcast.npz --no_features_scaling --split_type scaffold_balanced \
    --dataset_type classification --num_folds 10  --ensemble_size 1 --epochs 100 --metric auc --model_type dualmpnnplus \
    --batch_size 32 --depth 3 --hidden_size 3 --ffn_num_layers 4 --ffn_hidden_size 4 \
    --dropout 0.1 --weight_decay 1e-7 --bond_drop_rate 0 --save_dir <your saved dir>