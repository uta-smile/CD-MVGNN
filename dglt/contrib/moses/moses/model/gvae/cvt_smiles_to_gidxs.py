import os
import math
from multiprocessing import Process
import nltk
from nltk.grammar import Nonterminal, Production
import numpy as np
import pandas as pd
import argparse

class Grammar(object):
    def __init__(self, fpath):
        self.__init_grammar(fpath)

    def to_idxs(self, smiles):
        try:
            tokens = self.__tokenize(smiles)
            tree = list(self.cfg_parser.parse(tokens))[0]
            idxs = self.tree2idxs(tree)
            idxs_str = self.idxs2str(idxs)
        except:
            raise ValueError('failed to parse: ' + smiles)

        return idxs_str

    def to_idxs_batch(self, smiles_list):
        try:
            tokens_list = [self.__tokenize(smiles) for smiles in smiles_list]
            tree_list = [list(self.cfg_parser.parse(tokens))[0] for tokens in tokens_list]
            idxs_list = [self.tree2idxs(tree) for tree in tree_list]
            idxs_str_list = [self.idxs2str(idxs) for idxs in idxs_list]
        except:
            for smiles in smiles_list:
                self.to_idxs(smiles)

        return idxs_str_list

    def to_smiles(self, idxs):
        rule_idxs = [int(sub_str) for sub_str in idxs.split('|')]
        smiles, idx_base = self.__rule_idxs_to_smiles(rule_idxs, 0, 'smiles')

        return smiles

    def __rule_idxs_to_smiles(self, rule_idxs, idx_base, lhs_item):
        smiles = ''
        if idx_base >= len(rule_idxs):
            return smiles, idx_base

        rule_idx = rule_idxs[idx_base]
        lhs, rhs = [sub_str.strip() for sub_str in self.rules[rule_idx].split('->')]
        rhs_list = [sub_str.strip() for sub_str in rhs.split()]
        assert lhs == lhs_item
        idx_base += 1
        for rhs_item in rhs_list:
            if rhs_item == '\'\\\\\'':
                smiles += '\\'
            elif rhs_item.startswith('\'') and rhs_item.endswith('\''):
                smiles += rhs_item.strip('\'')
            else:
                item_str, idx_base = self.__rule_idxs_to_smiles(rule_idxs, idx_base, rhs_item)
                smiles += item_str

        return smiles, idx_base

    def __init_grammar(self, fpath):
        cfg_string = ''.join(list(open(fpath).readlines()))
        cfg_grammar = nltk.CFG.fromstring(cfg_string)
        self.cfg_parser = nltk.ChartParser(cfg_grammar)
        self.valid_tokens = cfg_grammar._lexical_index.keys()

        rule2idx = {}
        self.rules = []
        for idx, rule in enumerate(cfg_grammar.productions()):
            rule2idx[rule] = idx
            self.rules.append(str(rule))
        self.tree2idxs = lambda tree: [rule2idx[rule] for rule in tree.productions()]
        self.idxs2str = lambda idxs: '|'.join([str(idx) for idx in idxs])

    def __tokenize(self, smiles):
        result = []
        n = len(smiles)
        i = 0
        while i < n:
            j = i
            while j + 1 <= n and (smiles[i:j + 1] in self.valid_tokens
                                  or smiles[i:j + 1] == 'A'
                                  or smiles[i:j + 1] == 'M'
                                  or smiles[i:j + 1] == 'L'
                                  or smiles[i:j + 1] == 'R'
                                  or smiles[i:j + 1] == 'T'
                                  or smiles[i:j + 1] == 'Z'):
                j += 1
            if i == j:
                return None
            result.append(smiles[i:j])
            i = j

        return result

def cvt_smiles_to_idxs(grammar, smiles_list, idx_thread, csv_path=None):
    batch_size = 64
    nb_smpls = smiles_list.size
    nb_mbtcs = int(math.ceil(float(nb_smpls) / batch_size))
    print('[thread #%d] # of smiles: %d' % (idx_thread, nb_smpls))
    print('[thread #%d] # of mini-batches: %d' % (idx_thread, nb_mbtcs))
    idxs_list = []
    for idx_mbtc in range(nb_mbtcs):
        idx_smpl_beg = batch_size * idx_mbtc
        idx_smpl_end = min(idx_smpl_beg + batch_size, nb_smpls)
        idxs_list.extend(grammar.to_idxs_batch(smiles_list[idx_smpl_beg:idx_smpl_end]))
        print('[thread #%d] progress: %d / %d' % (idx_thread, idx_mbtc + 1, nb_mbtcs))

    # write SMILES and one-hot indices to *.csv file
    if csv_path is not None:
        smiles_n_idxs_df = pd.DataFrame(np.stack((smiles_list, idxs_list)).T, columns=['SMILES', 'IDXS'])
        smiles_n_idxs_df.to_csv(csv_path, header=False, index=False)
    return idxs_list

def verify_smiles_n_idxs(grammar, csv_path):
    csv_data = pd.read_csv(csv_path)
    nb_smpls = len(csv_data['SMILES'])
    for __ in range(100):
        idx_smpl = np.random.randint(nb_smpls)
        smiles_old = csv_data['SMILES'][idx_smpl]
        idxs_old = csv_data['IDXS'][idx_smpl]
        smiles_new = grammar.to_smiles(idxs_old)
        idxs_new = grammar.to_idxs(smiles_old)
        print('SMILES - old: ' + smiles_old)
        print('SMILES - new: ' + smiles_new)
        print('IDXS   - old: ' + idxs_old)
        print('IDXS   - new: ' + idxs_new)
        assert smiles_old == smiles_new and idxs_old == idxs_new
    print('verification passed')

def main():
    # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers",
                        type=int, default=1,
                        help="Number of workers")
    parser.add_argument("--grammar_path",
                        required=True, type=str,
                        help="The path fo grammar file")
    parser.add_argument("--input_csv",
                        type=str, required=True,
                        help="The path of processing csv file")
    parser.add_argument("--output_csv",
                        required=True, type=str,
                        help="The path of the output csv file")
    parser.add_argument("--tmp_dir",
                        type=str, default="./",
                        help="The path of the temp files")
    opts = parser.parse_args()
    nb_threads = opts.n_workers
    grammar_path = opts.grammar_path
    csv_path_src = opts.input_csv
    csv_path_dst = opts.output_csv
    csv_path_pattern_tmp = os.path.join(opts.tmp_dir, 'temp_n_gidxs_%d.csv')

    # create a grammar model for SMILES
    grammar = Grammar(grammar_path)

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
        args = (grammar, smiles_list_sel, idx_thread, csv_path_tmp)
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

    # verify pairs of SMILES & one-hot indices
    verify_smiles_n_idxs(grammar, csv_path_dst)

if __name__ == '__main__':
    main()
