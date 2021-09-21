import math

import numpy as np
import torch
from rdkit import Chem

from dglt.contrib.grover.mol2features import atom_to_vocab
from dglt.data.featurization import mol2graph


class GroverCollator(object):
    def __init__(self, shared_dict, vocab, args):
        self.args = args
        self.shared_dict = shared_dict
        self.vocab = vocab

    def random_mask(self, smiles_batch):
        # There is a zero padding.
        vocab_label = [0]
        percent = 0.15
        for smi in smiles_batch:
            mol = Chem.MolFromSmiles(smi)
            mlabel = [0] * mol.GetNumAtoms()
            n_mask = math.ceil(mol.GetNumAtoms() * percent)
            perm = np.random.permutation(mol.GetNumAtoms())[:n_mask]
            for p in perm:
                atom = mol.GetAtomWithIdx(int(p))
                mlabel[p] = self.vocab.stoi.get(atom_to_vocab(mol, atom), self.vocab.other_index)

            vocab_label.extend(mlabel)
        return vocab_label

    def __call__(self, batch):
        smiles_batch = [d.smiles for d in batch]
        batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args).get_components()
        vocab_label = torch.Tensor(self.random_mask(smiles_batch)).long()
        fgroup_label = torch.Tensor([d.features for d in batch]).float()
        # may be some mask here
        res = {"graph_input": batchgraph,
               "targets": {"av_task": vocab_label,
                           "fg_task": fgroup_label}
               }
        return res
