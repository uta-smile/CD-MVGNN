"""Computes and saves molecular features for a dataset."""

import os
import sys
from argparse import ArgumentParser, Namespace
from typing import List, Tuple

from dglt.data.featurization.mol2graph import MolGraph

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from dglt.data.featurization import load_features
from dglt.contrib.grover.mol2features import register_features_generator


def load_temp(temp_dir: str) -> Tuple[List[List[float]], int]:
    """
    Loads all features saved as .npz files in load_dir.

    Assumes temporary files are named in order 0.npz, 1.npz, ...

    :param temp_dir: Directory in which temporary .npz files containing features are stored.
    :return: A tuple with a list of molecule features, where each molecule's features is a list of floats,
    and the number of temporary files.
    """
    features = []
    temp_num = 0
    temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

    while os.path.exists(temp_path):
        features.extend(load_features(temp_path))
        temp_num += 1
        temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

    return features, temp_num


def filter_smiles(args: Namespace):
    args.coord = False
    args.bond_drop_rate = 0
    args.atom_messages = False
    args.no_cache = True

    mol_graphs = []
    fin = open(args.data_path, "r")
    fout = open(args.save_path, "w")
    fout.write(fin.readline())
    for l in fin:
        smiles = l.strip()
        try:
            mol_graph = MolGraph(smiles, args)
        except IndexError:
            print(smiles)
            continue
        fout.write(l)


if __name__ == '__main__':
    register_features_generator('fgtasklabel')
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to .npz file where features will be saved as a compressed numpy archive')
    args = parser.parse_args()

    filter_smiles(args)
