import torch
import numpy as np

from dglt.contrib.moses.moses.abstract_generator import AbstractGenerator


class GVAEGenerator(AbstractGenerator):
    """GVAE - SMILES generator."""

    def __init__(self, model, config, gen_config):
        """Constructor function."""

        super(GVAEGenerator, self).__init__(model, config, gen_config)

    def sample(self, nb_smpls, max_len):
        """Sample a list of SMILES sequences."""

        return self.model.sample(nb_smpls, max_len)

    def recon(self, x_idxs_list, max_len):
        """Reconstruct a list of SMILES sequences."""

        x_idxs_list = np.squeeze(x_idxs_list['IDXS'].values).tolist()
        return self.model.recon(x_idxs_list, max_len)

    def design(self, nb_smpls, properties, max_len):
        """Design a list of SMILES sequences that satisfy given property values."""

        return self.model.design(nb_smpls, properties, max_len)

    def recon_design(self, smiles_list_in, properties, max_len):
        """Design a list of SMILES sequences that satisfy given property values."""

        raise NotImplementedError
