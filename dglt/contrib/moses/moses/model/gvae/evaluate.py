import os
import pandas as pd
import rdkit
from rdkit import Chem

from dglt.contrib.moses.moses.metrics.utils import logP, SA, NP, QED, weight

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def count_valid_samples(smiles_list, calc_fn, val_min, val_max):
    cnt_match, cnt_valid = 0, 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        cnt_valid += 1 if mol is not None else 0
        try:
            val = calc_fn(mol)
            if val >= val_min and val <= val_max:
                cnt_match += 1
        except:
            pass
    print('# of valid samples: %d / %d' % (cnt_match, cnt_valid))

def eval_csv_file(path, range_list, target_list, tolerence=0.2):
    print('evaluating file: ' + path)
    smiles_list = pd.read_csv(path, usecols=['SMILES'], squeeze=True).astype(str).tolist()

    assert len(range_list) == len(target_list)
    val_min_list, val_max_list = [], []
    for (val_lbnd, val_ubnd), target in zip(range_list, target_list):
        val_min_list.append(val_lbnd + (target - tolerence) * (val_ubnd - val_lbnd))
        val_max_list.append(val_lbnd + (target + tolerence) * (val_ubnd - val_lbnd))

    count_valid_samples(smiles_list, logP,   val_min_list[0], val_max_list[0])
    count_valid_samples(smiles_list, SA,     val_min_list[1], val_max_list[1])
    count_valid_samples(smiles_list, NP,     val_min_list[2], val_max_list[2])
    count_valid_samples(smiles_list, QED,    val_min_list[3], val_max_list[3])
    count_valid_samples(smiles_list, weight, val_min_list[4], val_max_list[4])

### Main Entry ###

root_dir = '/data1/jonathan/Molecule.Generation/'
range_path = os.path.join(root_dir, 'AIPharmacist-data/props_min_n_max.txt')
csv_path_sample = os.path.join(root_dir, 'AIPharmacist-models/gvae_sample_3k.csv')
csv_path_design = os.path.join(root_dir, 'AIPharmacist-models/gvae_design_3k.csv')

with open(range_path, 'r') as i_file:
    range_list = [[float(x) for x in i_line.split()] for i_line in i_file]
#target_list = [0.9] * 5
#target_list = [0.1] * 5
#target_list = [0.6869, 0.1786, 0.2979, 0.4056, 0.2218]
#target_list = [0.3517, 0.6880, 0.4520, 0.5719, 0.2058]
target_list = [0.6588, 0.2467, 0.3372, 0.7682, 0.5377]

eval_csv_file(csv_path_sample, range_list, target_list)
eval_csv_file(csv_path_design, range_list, target_list)
