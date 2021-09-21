from abc import ABC
from abc import abstractmethod
from torch.utils.data import DataLoader

from dglt.contrib.moses.moses.utils import set_torch_seed_to_all_gens


class AbstractGenerator(ABC):
    """Abstract class for generating molecular structures, i.e. SMILES sequences."""

    def __init__(self, model, config, gen_config=None):
        """Constructor function.

        Args:
        * model: pre-trained model
        * config: configurations
        """

        self.model = model
        self.config = config
        self.gen_config = gen_config

    @property
    def n_workers(self):
        if self.gen_config is not None and 'n_workers' in self.gen_config:
            n_workers = self.gen_config.n_workers
        else:
            n_workers = self.config.n_workers
        return n_workers if n_workers != 1 else 0

    def get_collate_device(self, model):
        n_workers = self.n_workers
        return 'cpu' if n_workers > 0 else model.device

    def get_dataloader(self, model, data, collate_fn=None, shuffle=False):
        if collate_fn is None:
            collate_fn = self.get_collate_fn(model)
        if self.gen_config is not None and 'n_batch' in self.gen_config:
            n_batch = self.gen_config.n_batch
        else:
            n_batch = self.config.n_batch
        return DataLoader(data, batch_size=n_batch,
                          shuffle=shuffle, pin_memory= self.get_collate_device(model) == 'cpu',
                          num_workers=self.n_workers, collate_fn=collate_fn,
                          worker_init_fn=set_torch_seed_to_all_gens
                          if self.n_workers > 0 else None)

    def get_collate_fn(self, model):
        return None

    @abstractmethod
    def sample(self, nb_smpls, max_len):
        """Sample a list of SMILES sequences.

        Args:
        * nb_smpls: number of SMILES sequences to be sampled
        * max_len: maximal length of a SMILES sequence

        Returns:
        * smiles_list: list of SMILES sequences
        """
        pass

    @abstractmethod
    def recon(self, smiles_list_in, max_len):
        """Reconstruct a list of SMILES sequences.

        Args:
        * smiles_list_in: list of SMILES sequences to be reconstructed
        * max_len: maximal length of a SMILES sequence

        Returns:
        * smiles_list_out: list of SMILES sequences
        """
        pass

    @abstractmethod
    def design(self, nb_smpls, properties, max_len):
        """Design a list of SMILES sequences that satisfy given property values.

        Args:
        * nb_smpls: number of SMILES sequences to be designed
        * properties: dict of property values (can be a range) to be satisfied
        * max_len: maximal length of a SMILES sequence

        Returns:
        * smiles_list: list of SMILES sequences
        """
        pass

    @abstractmethod
    def recon_design(self, smiles_list_in, properties, max_len):
        """Reconstruct a list of SMILES sequences that satisfy
        given property values from smiles_list_in.

        Args:
        * smiles_list_in: list of SMILES sequences to be reconstructed
        * properties: dict of property values (can be a range) to be satisfied
        * max_len: maximal length of a SMILES sequence

        Returns:
        * smiles_list: list of SMILES sequences
        """
        pass
