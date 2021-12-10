CUDA_VISIBLE_DEVICES=0 python train.py --data_path <your data path>/bace.csv \
    --features_path <your data path>/bace.npz --no_features_scaling --split_type scaffold_balanced \
    --dataset_type classification --num_folds 10  --ensemble_size 1 --epochs 100 --metric auc --model_type dualmpnnplus \
    --init_lr 1.5e-4 --max_lr 1e-3 --final_lr 0.5e-4 --batch_size 32 --depth 4 --hidden_size 5 --ffn_num_layers 3 --ffn_hidden_size 2 \
    --dropout 0.2 --weight_decay 1e-10 --bond_drop_rate 0 --self_attention --attn_hidden 128 --attn_out 16 --save_dir <your saved dir>