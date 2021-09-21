import os
import numpy as np
import pandas as pd

root_dir = '/data1/jonathan/Molecule.Generation/AIPharmacist'
#csv_path_src = os.path.join(root_dir, 'data/test_n_cidxs.csv')
#rls_path_src = os.path.join(root_dir, 'data/crules.txt')
#csv_path_dst = os.path.join(root_dir, 'data/test_n_cidxs_rdc.csv')
#rls_path_dst = os.path.join(root_dir, 'data/crules_rdc.txt')
csv_path_src = os.path.join(root_dir, 'data/test_n_gidxs.csv')
rls_path_src = os.path.join(root_dir, 'data/grules.txt')
csv_path_dst = os.path.join(root_dir, 'data/test_n_gidxs_rdc.csv')
rls_path_dst = os.path.join(root_dir, 'data/grules_rdc.txt')

rules = []
with open(rls_path_src, 'r') as i_file:
    for i_line in i_file:
        rules.append(i_line.strip())
nb_rules = len(rules)

nb_occurs = np.zeros(nb_rules)
csv_data = pd.read_csv(csv_path_src)
for idxs_str in csv_data['IDXS']:
    idxs = [int(sub_str) for sub_str in idxs_str.split('|')]
    idxs_unq, cnts_unq = np.unique(idxs, return_counts=True)
    nb_occurs[idxs_unq] += cnts_unq

idxs_nnz = np.nonzero(nb_occurs)[0]
idx2idx_map = {int(x): np.nonzero(idxs_nnz == x)[0][0] for x in np.nditer(idxs_nnz)}
rules_nnz = [rules[idxs_nnz[idx]] for idx in range(idxs_nnz.size)]
with open(rls_path_dst, 'w') as o_file:
    o_file.write('\n'.join(rules_nnz))
print('reducing # of rules from %d to %d' % (nb_rules, idxs_nnz.size))

smiles_list = csv_data['SMILES']
idxs_list = []
for idxs_str in csv_data['IDXS']:
    idxs = [int(sub_str) for sub_str in idxs_str.split('|')]
    idxs = map(lambda x: idx2idx_map[x], idxs)
    idxs_list.append('|'.join([str(idx) for idx in idxs]))

# write SMILES and one-hot indices to *.csv file
smiles_n_idxs_df = pd.DataFrame({'SMILES': smiles_list, 'IDXS': idxs_list})
smiles_n_idxs_df.to_csv(csv_path_dst, index=False)
