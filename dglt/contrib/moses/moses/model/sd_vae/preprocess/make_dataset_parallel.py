#!/usr/bin/env python

from __future__ import print_function

import argparse

import dglt.contrib.moses.moses.model.sd_vae.utils.cfg_parser as parser
import numpy as np
from dglt.contrib.moses.moses.model.sd_vae.config import get_parser
from dglt.contrib.moses.moses.model.sd_vae.model.attribute_tree_decoder import create_tree_decoder
from dglt.contrib.moses.moses.model.sd_vae.model.mol_decoder import batch_make_att_masks, batch_make_att_masks_sparse
from dglt.contrib.moses.moses.model.sd_vae.model.tree_walker import OnehotBuilder
from dglt.contrib.moses.moses.model.sd_vae.preprocess.csv_saver import csv_writer
from dglt.contrib.moses.moses.model.sd_vae.utils.mol_util import MolUtil

from dglt.contrib.moses.moses.model.sd_vae.utils.mol_tree import AnnotatedTree2MolTree
from dglt.contrib.moses.moses.script_utils import read_smiles_csv

cmd_parser = argparse.ArgumentParser()
cmd_parser.add_argument('--smiles_file', help='list of smiles strings')
cmd_parser.add_argument('--model_save', default='.', help='result output root')
cmd_parser.add_argument('--n_workers', default=0, type=int, help='result output root')
cmd_args, _ = get_parser(cmd_parser).parse_known_args()
utils = MolUtil(cmd_args)

from joblib import Parallel, delayed
import h5py

def process_chunk(out_file, file_id, smiles_list):
    grammar = parser.Grammar(cmd_args.grammar_file)

    cfg_tree_list = []
    for smiles in smiles_list:
        ts = parser.parse(smiles, grammar)
        if not (isinstance(ts, list) and len(ts) == 1):
            print(smiles)
            #assert isinstance(ts, list) and len(ts) == 1
            continue

        n = AnnotatedTree2MolTree(ts[0], utils)
        cfg_tree_list.append(n)

    walker = OnehotBuilder(utils)
    tree_decoder = create_tree_decoder(utils)

    if cmd_args.save_as_csv:
        onehot, masks = batch_make_att_masks_sparse(cfg_tree_list, utils, tree_decoder,
                                                    walker, dtype=np.byte)
        csv_writer(onehot, masks, "%s_%d.csv" %(out_file, file_id), smiles_list)
        return None

    onehot, masks = batch_make_att_masks(cfg_tree_list, utils, tree_decoder, walker, dtype=np.byte)
    return (onehot, masks)

def run_job(L):
    chunk_size = 10000

    f_smiles = '.'.join(cmd_args.smiles_file.split('/')[-1].split('.')[0:-1])
    out_file = '%s/%s-%d' % (cmd_args.model_save, f_smiles, cmd_args.skip_deter)

    list_binary = Parallel(n_jobs=cmd_args.n_workers, verbose=50)(
        delayed(process_chunk)(out_file, start // chunk_size, L[start: start + chunk_size])
        for start in range(0, len(L), chunk_size)
    )

    if not cmd_args.save_as_csv:
        all_onehot = np.zeros((len(L), cmd_args.max_decode_steps, utils.DECISION_DIM), dtype=np.byte)
        all_masks = np.zeros((len(L), cmd_args.max_decode_steps, utils.DECISION_DIM), dtype=np.byte)

        for start, b_pair in zip( range(0, len(L), chunk_size), list_binary ):
            all_onehot[start: start + chunk_size, :, :] = b_pair[0]
            all_masks[start: start + chunk_size, :, :] = b_pair[1]

        out_file = out_file + ".h5"
        h5f = h5py.File(out_file, 'w')
        h5f.create_dataset('x', data=all_onehot)
        h5f.create_dataset('masks', data=all_masks)
        h5f.close()

if __name__ == '__main__':

    smiles_list = np.squeeze(read_smiles_csv(cmd_args.smiles_file).values).tolist()

    run_job(smiles_list)
    


