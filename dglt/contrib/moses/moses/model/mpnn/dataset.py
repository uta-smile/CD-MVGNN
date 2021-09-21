import numpy as np

from dglt.contrib.moses.moses.data.mpnn.data.data import MoleculeDatapoint, MoleculeDataset
from dglt.contrib.moses.moses.model.gvae.cvt_smiles_to_gidxs import Grammar
from joblib import Parallel, delayed
from tqdm import tqdm

class MPGVAEDataset(MoleculeDataset):
    """A MoleculeDataset contains a list of molecules and their associated features and targets."""

    def __init__(self, data, config):
        n_workers = 1 if config.n_workers <= 0 else config.n_workers

        # split data into smiles, features, and idxes
        smiles = data['SMILES'].squeeze().tolist()
        features = [np.array(list(map(float, _.split('|'))))
                    for _ in data['FEATURES'].squeeze().tolist()]

        # Process MPNN data
        super(MPGVAEDataset, self).__init__(list(Parallel(n_jobs=n_workers)(
            delayed(MoleculeDatapoint)([smile], features=feat, args=config)
            for smile, feat in tqdm(zip(smiles, features), desc="Loading mpnn data:"))))

        # Process GVAE data
        self.grammar = Grammar(config.rule_path)
        self.gvae_data = data['IDXS'].squeeze().tolist()

    def __getitem__(self, idx):
        """
        Gets one or more MoleculeDatapoints via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A pair of a MoleculeDatapoint and GVAE rules.
        """
        x = self.data[idx]
        if self.args is not None and np.random.random() < self.args.aug_rate:
            self.randomize_smiles(x)
            gvae_x = self.grammar.to_idxs(x.smiles)
        else:
            gvae_x = self.gvae_data[idx]

        return x, gvae_x
