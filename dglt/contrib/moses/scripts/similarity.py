import pandas as pd
import numpy as np
import argparse

import rdkit
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from joblib import Parallel, delayed
from tqdm import tqdm
rdkit.RDLogger.DisableLog('rdApp.*')

from dglt.contrib.moses.moses.utils import valid_smiles

parser = argparse.ArgumentParser("Compute the similarity")
parser.add_argument('--test_load',
                    type=str,
                    help='The test smiles file to be compared')
parser.add_argument('--recon_load',
                    type=str, default=None, required=True,
                    help='The generated smiles file')
parser.add_argument('--ref_load',
                    required=True, type=str, nargs='+', metavar='FILE',
                    help='The reference smiles files for generated smiles file. '
                         'It usually includes the train smiles and valid smiles')
parser.add_argument('--result_save',
                    type=str, default=None,
                    help='The path to save the results. '
                         'Default is recon filename+_sorted.csv')
parser.add_argument('--gen_ref',
                    type=str, default=None, nargs='?', const='-', metavar='FILE',
                    help='The smiles file that generated from. '
                         'If no FILE is provided, the last ref_load will be used.')
parser.add_argument('--n_workers',
                    type=int, default=-1,
                    help='Number of workers')
opts = parser.parse_args()

if opts.test_load is not None:
    smiles_data = pd.read_csv(opts.test_load, usecols=['SMILES']).values
else:
    smiles_data = None
recon_data = pd.read_csv(opts.recon_load)
if 'Similarity_Train' in recon_data:
    train_res = recon_data['Similarity_Train'].values.flatten()
    recon_data.drop(columns='Similarity_Train', inplace=True)
else:
    train_res = None
recon_data = recon_data.values
if opts.gen_ref is None:
    recon_data = np.array([list(set(recon_data.flatten().tolist()))])

ref_data = []
for f in opts.ref_load:
    ref_data.append(pd.read_csv(f, usecols=['SMILES']).values)
train_data = np.vstack(ref_data)
if opts.gen_ref is None:
    valid_data = None
elif opts.gen_ref == '-':
    valid_data = pd.read_csv(opts.ref_load[-1], usecols=['SMILES']).values
else:
    valid_data = pd.read_csv(opts.gen_ref, usecols=['SMILES']).values

save_path = opts.result_save
if save_path is None:
    save_path = opts.recon_load.strip(".csv") + "_sorted.csv"

def get_test_train(sm1):
    r = []
    r_id = []
    fp1 = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(sm1), 2)
    for i, sm1 in enumerate(recon_data):
        for sm2 in sm1:
            if not isinstance(sm2, float) and Chem.MolFromSmiles(sm2) is not None:
                fp2 = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(sm2), 2)
                r.append(DataStructs.TanimotoSimilarity(fp1, fp2))
                r_id.append(i)
            else:
                r.append(0.0)
                r_id.append(0)
    return r, r_id

if smiles_data is not None:
    res = Parallel(n_jobs=opts.n_workers)(delayed(get_test_train)(_) for _ in tqdm(
        smiles_data[:, 0],
        desc='Computing similarity for test data'
    ))
    res, res_id = list(zip(*res))
else:
    res, res_id = None, None

def get_train_sim(sm1):
    if not isinstance(sm1, float) and Chem.MolFromSmiles(sm1) is not None:
        fp1 = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(sm1), 2)
        sim_train = [DataStructs.TanimotoSimilarity(
            fp1,
            AllChem.GetMorganFingerprint(
                Chem.MolFromSmiles(_),
                2
            )) if Chem.MolFromSmiles(_) is not None else 0.0 for _ in train_data[:, 0]]
        return np.max(sim_train)

    return 0.0

if train_res is None:
    train_data = np.array([[_] for _ in train_data[:,0]if valid_smiles(_) is not None])
    train_res = Parallel(n_jobs=opts.n_workers)(delayed(get_train_sim)(_) for _ in tqdm(
        recon_data.flatten(),
        desc='Computing similarity for train data'
    ))
train_res_arr = np.array(train_res)

if res is not None:
    res_arr = np.array(res)
    res_id_arr = np.array(res_id)
    sim = np.max(res_arr, axis=0).flatten()
    sim_id = np.argmax(res_arr, axis=0)
    sim_test = smiles_data[sim_id].flatten()

    res_arr_ex = res_arr.copy()
    res_arr_ex[:, train_res_arr == 1] = 0
    sim_test_valid = np.max(res_arr_ex, axis=1).flatten()
    sim_test_valid_id = np.argmax(res_arr_ex, axis=1)

    thres = [_ / 10.0 for _ in range(10)]
    print("|" + '|'.join(map(str,thres)) + "|")
    print("|" + '|'.join(map(lambda s:str(np.sum(sim > s)), thres)) + "|")
    print("|" + '|'.join(map(lambda s:str(np.sum((sim > s) & (train_res_arr != 1))), thres)) + "|")
    print("object_value=%0.5f" % (np.max(sim[train_res_arr != 1])))

results = []
results_cols = []
if res is not None:
    results.append(sim_test)
    results_cols.append('Test_SMILES')
results.append(recon_data.flatten())
results_cols.append('Recon_SMILES')
if res is not None:
    results.append(sim)
    results_cols.append('Similarity')
results.append(train_res)
results_cols.append('Similarity_Train')
if valid_data is not None and res is not None:
    sim_valid = valid_data[res_id_arr[sim_id,range(sim_id.shape[0])]].flatten()
    results.append(sim_valid)
    results_cols.append('Orig_SMILES')

output_data = pd.DataFrame(np.stack(results).T, columns=results_cols)

output_data.sort_values('Similarity', ascending=False)\
    .to_csv(save_path)

results = []
results_cols = []
if res is not None:
    results.append(smiles_data.flatten())
    results_cols.append('Test_SMILES')
    recon_data = recon_data.flatten()
    sim_test_smiles = recon_data[sim_test_valid_id]
    results.append(sim_test_smiles)
    results_cols.append('Recon_SMILES')
    results.append(sim_test_valid)
    results_cols.append('Similarity')
    sim_train = train_res_arr[sim_test_valid_id]
    results.append(sim_train)
    results_cols.append('Similarity_Train')

output_data = pd.DataFrame(np.stack(results).T, columns=results_cols)

output_data.sort_values('Similarity', ascending=False)\
    .to_csv(save_path.strip('.csv') + '_test.csv')
