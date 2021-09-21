#!/usr/bin/env python

from __future__ import print_function

import argparse
from rdkit import Chem

import h5py
from dglt.contrib.moses.moses.model.sd_vae.config import get_parser
from dglt.contrib.moses.moses.model.sd_vae.model.mol_decoder import batch_make_att_masks, batch_make_att_masks_sparse
from dglt.contrib.moses.moses.model.sd_vae.preprocess.csv_saver import csv_writer
from dglt.contrib.moses.moses.model.sd_vae.utils import cfg_parser as parser
from dglt.contrib.moses.moses.model.sd_vae.utils.mol_util import MolUtil
from dglt.contrib.moses.moses.model.sd_vae.utils.mol_tree import get_smiles_from_tree
from tqdm import tqdm

from dglt.contrib.moses.moses.model.sd_vae.utils.mol_tree import AnnotatedTree2MolTree
from dglt.contrib.moses.moses.script_utils import read_smiles_csv

from pdb import set_trace as cp
# from past.builtins import range

cmd_parser = argparse.ArgumentParser()
cmd_parser.add_argument('--smiles_file', help='list of smiles strings')
cmd_parser.add_argument('--model_save', default='.', help='result output root')
cmd_args, _ = get_parser(cmd_parser).parse_known_args()
utils = MolUtil(cmd_args)
print("Rule size %d" % utils.DECISION_DIM)

def parse_smiles_with_cfg(smiles_file):
    grammar = parser.Grammar(cmd_args.grammar_file)

    cfg_tree_list = []
    invalid_rows = []
    for row in tqdm(smiles_file.values.flatten()):
        smiles = row.strip()
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            invalid_rows.append(row)
            continue
        smiles = Chem.MolToSmiles(m, isomericSmiles=True)
        if smiles is None:
            invalid_rows.append(row)
            continue
        ts = parser.parse(smiles, grammar)
        if not (isinstance(ts, list) and len(ts) == 1):
            invalid_rows.append(row)
            #assert isinstance(ts, list) and len(ts) == 1
            continue
        n = AnnotatedTree2MolTree(ts[0], utils)
        recon_smiles = get_smiles_from_tree(n)
        assert smiles == recon_smiles
        cfg_tree_list.append(n)
    
    print(invalid_rows)

    return cfg_tree_list

def run_job(smiles_list):
    cfg_tree_list = parse_smiles_with_cfg(smiles_list)

    if cmd_args.save_as_csv:
        all_true_binary, all_rule_masks = batch_make_att_masks_sparse(cfg_tree_list, utils)
        print(len(all_true_binary), len(all_true_binary[0]), all_rule_masks.shape)

        f_smiles = '.'.join(cmd_args.smiles_file.split('/')[-1].split('.')[0:-1])
        out_file = '%s/%s.csv' % (cmd_args.model_save, f_smiles)
        csv_writer(all_true_binary, all_rule_masks, out_file, smiles_list)
    else:
        all_true_binary, all_rule_masks = batch_make_att_masks(cfg_tree_list, utils)
        print(all_true_binary.shape, all_rule_masks.shape)

        f_smiles = '.'.join(cmd_args.smiles_file.split('/')[-1].split('.')[0:-1])
        out_file = '%s/%s.h5' % (cmd_args.model_save, f_smiles)
        h5f = h5py.File(out_file, 'w')

        h5f.create_dataset('x', data=all_true_binary)
        h5f.create_dataset('masks', data=all_rule_masks)
        h5f.close()


if __name__ == '__main__':
    import sys, traceback, pdb
    try:
        smiles_list = read_smiles_csv(cmd_args.smiles_file)
        run_job(smiles_list)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

