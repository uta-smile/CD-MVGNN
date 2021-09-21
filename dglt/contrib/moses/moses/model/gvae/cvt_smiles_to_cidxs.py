import os
import math
from multiprocessing import Process
import nltk
from nltk.grammar import Nonterminal, Production
import numpy as np
import pandas as pd

def cvt_smiles_to_idxs(smiles_list, idx_thread, csv_path, chr_path):
    batch_size = 64

    chars = []
    with open(chr_path, 'r') as i_file:
        chars = [i_line.strip() for i_line in i_file]
    map_fn = lambda x: chars.index(x)
    def __to_idxs_batch(smiles_list):
        return ['|'.join([str(idx) for idx in map(map_fn, smiles)]) for smiles in smiles_list]

    nb_smpls = smiles_list.size
    nb_mbtcs = int(math.ceil(float(nb_smpls) / batch_size))
    print('[thread #%d] # of smiles: %d' % (idx_thread, nb_smpls))
    print('[thread #%d] # of mini-batches: %d' % (idx_thread, nb_mbtcs))
    idxs_list = []
    for idx_mbtc in range(nb_mbtcs):
        idx_smpl_beg = batch_size * idx_mbtc
        idx_smpl_end = min(idx_smpl_beg + batch_size, nb_smpls)
        idxs_list.extend(__to_idxs_batch(smiles_list[idx_smpl_beg:idx_smpl_end]))
        print('[thread #%d] progress: %d / %d' % (idx_thread, idx_mbtc + 1, nb_mbtcs))

    # write SMILES and one-hot indices to *.csv file
    smiles_n_idxs_df = pd.DataFrame({'SMILES': smiles_list, 'IDXS': idxs_list})
    smiles_n_idxs_df.to_csv(csv_path, header=False, index=False)

def main():
    # configuration
    nb_threads = 16
    root_dir = '/data1/jonathan/Molecule.Generation/AIPharmacist'
    chr_path = os.path.join(root_dir, 'data/crules.txt')
    csv_path_src = os.path.join(root_dir, 'data/train.csv')
    csv_path_dst = os.path.join(root_dir, 'data/train_n_cidxs.csv')
    csv_path_pattern_tmp = os.path.join(root_dir, 'data/train_n_cidxs_%d.csv')

    # load SMILES sequences from *.csv file
    csv_data = pd.read_csv(csv_path_src)
    smiles_list = csv_data['SMILES']

    # convert SMILES sequences into one-hot indices
    nb_smpls = len(smiles_list)
    nb_smpls_per_thread = int(math.ceil(float(nb_smpls) / nb_threads))
    procs = []
    for idx_thread in range(nb_threads):
        idx_smpl_beg = nb_smpls_per_thread * idx_thread
        idx_smpl_end = min(idx_smpl_beg + nb_smpls_per_thread, nb_smpls)
        smiles_list_sel = smiles_list[idx_smpl_beg:idx_smpl_end]
        csv_path_tmp = csv_path_pattern_tmp % idx_thread
        args = (smiles_list_sel, idx_thread, csv_path_tmp, chr_path)
        proc = Process(target=cvt_smiles_to_idxs, args=args)
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()

    # merge multiple *.csv files into one
    with open(csv_path_dst, 'w') as o_file:
        o_file.write('SMILES,IDXS\n')
        for idx_thread in range(nb_threads):
            csv_path_tmp = csv_path_pattern_tmp % idx_thread
            with open(csv_path_tmp, 'r') as i_file:
                o_file.write(i_file.read())
            os.remove(csv_path_tmp)

if __name__ == '__main__':
    main()
