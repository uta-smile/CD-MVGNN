## Data
Datasets used are from MoleculeNet [1]. We also include the csv files of the data in the data folder.
[1] Wu, Zhenqin, Bharath Ramsundar, Evan N. Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S. Pappu, Karl Leswing, and Vijay Pande. "MoleculeNet: a benchmark for molecular machine learning." Chemical science 9, no. 2 (2018): 513-530.

## Training example

```python
### Generate rdkit features

python scripts/save_features.py --data_path bbbp.csv --features_generator rdkit_2d_normalized --save_path bbbp.npz --restart

### train CD-MVGNN model
CUDA_VISIBLE_DEVICES=0 python train.py --data_path ../data/bbbp.csv \
--features_path ../data/bbbp.npz --no_features_scaling \
--split_type scaffold_balanced --dataset_type classification --epochs 40 --metric auc \
--model_type dualmpnnplus --init_lr 1.5e-4 --max_lr 1e-3 --final_lr 1.5e-4 --batch_size 32 --depth 4 \
--hidden_size 4 --ffn_num_layers 4 --ffn_hidden_size 2 --dropout 0 --weight_decay 1e-7 
```
The training settings for each dataset is also provided in the scripts folder, which can be run by
```
bash <dataset>.sh
```
## Conda environment
The environment can be created by 
```
conda env create -f environment.yaml
```