import torch

from dglt.contrib.moses.moses.abstract_generator import AbstractGenerator

import numpy as np
from tqdm import tqdm

from dglt.contrib.moses.moses.interfaces import MosesTrainer
from dglt.contrib.moses.moses.model.sd_vae.utils import cfg_parser as parser
from .preprocess.dataset import SDVAEDataset
from .utils.mol_util import MolUtil


class SDVAEGenerator(AbstractGenerator):
    def __init__(self, model, config, gen_config=None):

        super(SDVAEGenerator, self).__init__(model, config, gen_config)
        self.utils = MolUtil(self.config)
        self.grammar = parser.Grammar(self.config.grammar_file)

    def get_collate_fn(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            onehot, _, smiles = zip(*data)
            if model.onehot:
                onehot = torch.tensor(np.stack(onehot), dtype=torch.float, device=device)
            else:
                order = np.argsort([len(_) for _ in onehot])[::-1]
                onehot = [torch.tensor(onehot[_], dtype=torch.long, device=device) for _ in
                          order]
                smiles = [smiles[_] for _ in order]

            return onehot, smiles

        return collate

    def _recon_epoch(self, model, input_batch, max_len, device, use_random):
        if model.onehot:
            x_inputs = input_batch[0].permute(0, 2, 1).to(device)
        else:
            x_inputs = tuple(data.to(device) for data in input_batch[0])

        # Reconstruct
        decode_results = model.recon(x_inputs, max_len,
                                     encode_times=self.gen_config.encode_times,
                                     decode_times=self.gen_config.decode_times,
                                     use_random=use_random)
        postfix = {}
        postfix['junk'] = np.sum(np.vectorize(lambda dec: dec.startswith('JUNK'))(decode_results))
        postfix['total'] = decode_results.size
        if input_batch[1][0] is not None:
            postfix["acc"] = np.sum([_0 == _1 for _0, _1 in zip(input_batch[1], decode_results)])
        return decode_results, postfix

    def _recon(self, model, recon_loader, max_len, use_random=True):
        device = model.device
        if use_random:
            model.train()
        else:
            model.eval()
        samples = []
        postfix = {'junk': 0,
                   'total': 0,
                   'acc': 0}

        tqdm_data = tqdm(recon_loader, desc='Reconstructing samples')
        for batch in tqdm_data:
            current_samples, post = self._recon_epoch(model, batch, max_len, device, use_random)
            samples.append(current_samples)
            for key, value in post.items():
                postfix[key] += value
            rate = postfix["junk"] * 1.0 / postfix["total"]
            postfix_list = [f"junk={postfix['junk']:d}",
                            f"total={postfix['total']:d}",
                            f"rate={rate:.5f}"]
            if 'acc' in post:
                accuracy = postfix['acc'] * 1.0 / postfix["total"]
                postfix_list += [f"correct={postfix['acc']:d}",
                                 f"accuracy={accuracy:.5f}"]
            tqdm_data.set_postfix_str(' '.join(postfix_list))

        return np.vstack(samples)

    def recon(self, smiles_list_in, max_len):
        recon_data = smiles_list_in['SDVAE'].values
        smiles_data = smiles_list_in['SMILES'].values

        recon_loader = self.get_dataloader(self.model, SDVAEDataset(recon_data, self.config,
                                                                    smiles=smiles_data,
                                                                    grammar=self.grammar,
                                                                    onehot=self.model.onehot))

        smiles_list =  self._recon(self.model, recon_loader, max_len,
                                   use_random=self.gen_config.use_random)

        return smiles_list

    def sample(self, nb_smpls, max_len):
        return self.model.sample(nb_smpls, max_len)

    def design(self, nb_smpls, properties, max_len):

        raise NotImplementedError

    def recon_design(self, smiles_list_in, properties, max_len):
        """Design a list of SMILES sequences that satisfy given property values."""

        raise NotImplementedError

            # def _n_epoch(self):
    #     return sum(
    #         self.config.lr_n_period * (self.config.lr_n_mult ** i)
    #         for i in range(self.config.lr_n_restarts)
    #     )

