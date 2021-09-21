import torch
from scipy import linalg
from dglt.data.featurization import mol2graph

class MolCollator(object):
    def __init__(self, shared_dict, args):
        self.args = args
        self.shared_dict = shared_dict

    def __call__(self, batch):
        smiles_batch = [d.smiles for d in batch]
        features_batch = [d.features for d in batch]
        target_batch = [d.targets for d in batch]
        batch_mol_graph = mol2graph(smiles_batch, self.shared_dict, self.args)
        batch = batch_mol_graph.get_components()
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])
        if self.args.dataset_type == 'multiclass':
            targets = targets.long()
            targets = targets.squeeze()
        return smiles_batch, batch, features_batch, mask, targets
