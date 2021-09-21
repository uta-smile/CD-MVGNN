import numpy as np
from functools import wraps
from tqdm import tqdm
import torch

from dglt.contrib.moses.moses.abstract_generator import AbstractGenerator
from dglt.contrib.moses.moses.model.vae.dataset import VAEDesignDataset


class VAEGenerator(AbstractGenerator):
    """VAE - SMILES generator."""

    def __init__(self, model, config, gen_config=None):
        """Constructor function."""

        super(VAEGenerator, self).__init__(model, config, gen_config)

    def sample(self, nb_smpls, max_len):
        """Sample a list of SMILES sequences."""

        return self.model.sample(nb_smpls, max_len)

    def get_collate_fn_recon(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device=device)
                       for string in data]

            return tensors, data
        return collate

    def _recon_epoch(self, model, input_batch, max_len, device):
        x_inputs = [_.to(device) for _ in input_batch[0]]

        # Reconstruct
        decode_results = model.recon(x_inputs, max_len=max_len,
                                     encode_times=self.gen_config.encode_times,
                                     decode_times=self.gen_config.decode_times)
        postfix = {}
        postfix['total'] = decode_results.size
        postfix["acc"] = np.sum([_0 == _1 for _0, _1 in zip(input_batch[1], decode_results)])
        return decode_results, postfix

    def _recon(self, model, recon_loader, max_len):
        device = model.device
        model.eval()
        samples = []
        postfix = {'total': 0,
                   'acc': 0}

        tqdm_data = tqdm(recon_loader, desc='Reconstructing samples')
        for batch in tqdm_data:
            current_samples, post = self._recon_epoch(model, batch, max_len, device)
            samples.append(current_samples)
            for key, value in post.items():
                postfix[key] += value
            postfix_list = [f"total={postfix['total']:d}"]
            if 'acc' in post:
                accuracy = postfix['acc'] * 1.0 / postfix["total"]
                postfix_list += [f"correct={postfix['acc']:d}",
                                 f"accuracy={accuracy:.5f}"]
            tqdm_data.set_postfix_str(' '.join(postfix_list))

        return np.vstack(samples)

    def recon(self, smiles_list_in, max_len):
        recon_data = smiles_list_in['SMILES']

        recon_loader = self.get_dataloader(self.model, recon_data, shuffle=False,
                                           collate_fn=self.get_collate_fn_recon(self.model))

        smiles_list = self._recon(self.model, recon_loader, max_len)

        return smiles_list

    def get_collate_fn_design(self, model):
        device = self.get_collate_device(model)

        @wraps(device)
        def collate(data):
            return torch.tensor(np.stack(data).astype(np.float), dtype=torch.float, device=device)

        return collate

    def design(self, nb_smpls, properties, max_len):
        """Design a list of SMILES sequences that satisfy given property values."""

        if self.config.auto_norm:
            min_v = self.model.min.data.cpu().numpy()
            max_v = self.model.max.data.cpu().numpy()
        else:
            min_v = None
            max_v = None
        design_data = self.get_dataloader(self.model,
                                          VAEDesignDataset(properties, nb_smpls,
                                                           min_v=min_v, max_v=max_v),
                                          collate_fn=self.get_collate_fn_design(self.model))
        tqdm_data = tqdm(design_data, desc='Generating designed SMILES sequences')

        smiles_list = []
        for input_batch in tqdm_data:
            input_batch = input_batch.to(self.model.device)
            current_list = self.model.design(input_batch, max_len)
            smiles_list.extend(current_list)

        return smiles_list

    def recon_design(self, smiles_list_in, properties, max_len):
        """Design a list of SMILES sequences that satisfy given property values."""

        raise NotImplementedError

        if self.config.auto_norm:
            min_v = self.model.min.data.cpu().numpy()
            max_v = self.model.max.data.cpu().numpy()
        else:
            min_v = None
            max_v = None
        design_data = self.get_dataloader(self.model,
                                          VAEDesignDataset(properties, 1,
                                                           min_v=min_v, max_v=max_v))
        tqdm_data = tqdm(design_data, desc='Generating designed SMILES sequences')

        smiles_list = []
        for input_batch in tqdm_data:
            input_batch = input_batch.to(self.model.device)
            current_list = self.model.recon(smile_list_in, input_batch, max_len)
            smiles_list.extend(current_list)

        return smiles_list
