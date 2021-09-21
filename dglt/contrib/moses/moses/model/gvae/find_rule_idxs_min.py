import pandas as pd

def find_rule_idxs_min(csv_path):
    rule_idxs_min = set()
    rule_idxs_list = pd.read_csv(csv_path, usecols=['IDXS'], squeeze=True).astype(str).tolist()
    for rule_idxs in rule_idxs_list:
        rule_idxs_min |= set([int(x) for x in rule_idxs.split('|')])
    print(rule_idxs_min)

    return rule_idxs_min

def cvt_rule_idxs(csv_path_old, csv_path_new, rule_idxs_map_fn):
    csv_data_old = pd.read_csv(csv_path_old)

    rule_idxs_list = []
    for rule_idxs_old in csv_data_old['IDXS']:
        rule_idxs_old = [int(x) for x in rule_idxs_old.split('|')]
        rule_idxs_new = list(map(rule_idxs_map_fn, rule_idxs_old))
        rule_idxs_new = '|'.join(['%d' % x for x in rule_idxs_new])
        rule_idxs_list.append(rule_idxs_new)

    csv_data_new = pd.DataFrame({'SMILES': csv_data_old['SMILES'], 'IDXS': rule_idxs_list})
    csv_data_new.to_csv(csv_path_new, index=False)

### MAIN ENTRY ###

CSV_PATH_TRN_ALL = '../../data/train_n_gidxs.csv'
CSV_PATH_TRN_MIN = '../../data/train_n_gidxs_min.csv'
CSV_PATH_TST_ALL = '../../data/test_n_gidxs.csv'
CSV_PATH_TST_MIN = '../../data/test_n_gidxs_min.csv'
GRULES_PATH_ALL = './grules.txt'
GRULES_PATH_MIN = './grules_min.txt'

rule_idxs_min = set()
rule_idxs_min |= find_rule_idxs_min(CSV_PATH_TRN_ALL)
rule_idxs_min |= find_rule_idxs_min(CSV_PATH_TST_ALL)

rule_idxs_min = list(rule_idxs_min)
rule_idxs_min.sort()
rule_idxs_map = {rule_idx: idx for idx, rule_idx in enumerate(rule_idxs_min)}
rule_idxs_map_fn = lambda x: rule_idxs_map[x]
print(rule_idxs_map)

cvt_rule_idxs(CSV_PATH_TRN_ALL, CSV_PATH_TRN_MIN, rule_idxs_map_fn)
cvt_rule_idxs(CSV_PATH_TST_ALL, CSV_PATH_TST_MIN, rule_idxs_map_fn)

with open(GRULES_PATH_ALL, 'r') as i_file:
    with open(GRULES_PATH_MIN, 'w') as o_file:
        for idx, i_line in enumerate(i_file):
            if idx in rule_idxs_map:
                o_file.write(i_line)
