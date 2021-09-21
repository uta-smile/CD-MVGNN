import numpy as np
from functools import wraps
from tqdm import tqdm
import torch

from dglt.contrib.moses.moses.abstract_generator import AbstractGenerator
from dglt.contrib.moses.moses.model.vae.dataset import VAEDesignDataset


class ORGANGenerator(AbstractGenerator):
    """ORGAN - SMILES generator."""

    def __init__(self, model, config, gen_config=None):
        """Constructor function."""

        super(ORGANGenerator, self).__init__(model, config, gen_config)

    def sample(self, nb_smpls, max_len):
        """Sample a list of SMILES sequences."""

        return self.model.sample(nb_smpls, max_len)

    def recon(self, smiles_list_in, max_len):
        """Reconstruct a list of SMILES sequences from reference SMILES."""

        raise NotImplementedError

    def design(self, nb_smpls, properties, max_len):
        """Design a list of SMILES sequences that satisfy given property values."""

        raise NotImplementedError

    def recon_design(self, smiles_list_in, properties, max_len):
        """Design a list of SMILES sequences that satisfy given property values."""

        raise NotImplementedError
