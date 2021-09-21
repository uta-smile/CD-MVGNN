import torch

from dglt.contrib.moses.moses.abstract_generator import AbstractGenerator


class JTVAEGenerator(AbstractGenerator):
    """VAE - SMILES generator."""

    def __init__(self, model, config, gen_config=None):
        """Constructor function."""

        super(JTVAEGenerator, self).__init__(model, config, gen_config)

    def sample(self, nb_smpls, max_len):
        """Sample a list of SMILES sequences."""

        return self.model.sample(nb_smpls, max_len)

    def recon(self, nb_smpls, max_len):
        """Reconstruct a list of SMILES sequences."""

        raise NotImplementedError

    def design(self, nb_smpls, properties, max_len):
        """Design a list of SMILES sequences that satisfy given property values."""

        raise NotImplementedError

    def recon_design(self, smiles_list_in, properties, max_len):
        """Design a list of SMILES sequences that satisfy given property values."""

        raise NotImplementedError
