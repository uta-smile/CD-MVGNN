#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dglt.contrib.moses.moses.utils import set_torch_seed_to_all_gens
from dglt.contrib.moses.moses.model.sd_vae.utils.mol_util import MolUtil
from dglt.contrib.moses.moses.model.sd_vae.model.mol_decoder import PerpCalculator, StateDecoder
from dglt.contrib.moses.moses.model.sd_vae.model.mol_encoder import CNNEncoder
from dglt.contrib.moses.moses.model.sd_vae.preprocess.dataset import SDVAEGenerater

# def parse_smiles_with_cfg(smiles_file, grammar_file):
#     grammar = parser.Grammar(grammar_file)
#
#     smiles_list = []
#     cfg_tree_list = []
#     annotated_trees = []
#     with open(smiles_file, 'r') as f:
#         for row in tqdm(f):
#             smiles = row.strip()
#             smiles_list.append(smiles)
#             ts = parser.parse(smiles, grammar)
#             assert isinstance(ts, list) and len(ts) == 1
#             annotated_trees.append(ts[0])
#             n = AnnotatedTree2MolTree(ts[0])
#             cfg_tree_list.append(n)
#             st = get_smiles_from_tree(n)
#
#             assert st == smiles
#
#     return (smiles_list, cfg_tree_list, annotated_trees)


def get_encoder(cmd_args, utils):
    if cmd_args.encoder_type == 'cnn':
        return CNNEncoder(max_len=cmd_args.max_decode_steps, latent_dim=cmd_args.d_z,
                          DECISION_DIM=utils.DECISION_DIM)
    else:
        raise ValueError('unknown encoder type %s' % cmd_args.encoder_type)


class MolAutoEncoder(nn.Module):
    def __init__(self, vocab, cmd_args):
        super(MolAutoEncoder, self).__init__()
        print('using auto encoder')
        self.vocab = vocab
        self.latent_dim = cmd_args.d_z
        utils = MolUtil(cmd_args)
        self.encoder = get_encoder(cmd_args, utils)
        self.state_decoder = StateDecoder(max_len=cmd_args.max_decode_steps,
                                          latent_dim=cmd_args.d_z,
                                          DECISION_DIM=utils.DECISION_DIM)
        self.perp_calc = PerpCalculator()
        self.onehot = True


    def forward(self, x_inputs, true_binary, rule_masks):
        z, _ = self.encoder(x_inputs)

        raw_logits = self.state_decoder(z)
        perplexity = self.perp_calc(true_binary, rule_masks, raw_logits)

        return (perplexity)

    @property
    def device(self):
        return next(self.parameters()).device

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return torch.randn(n_batch, self.q_mu.out_features,
                           device=self.x_emb.weight.device)

    def sample(self, n_batch, max_len=None, z=None, temp=1.0, sample_times=100, valid=True):
        """Generating n_batch samples in eval mode (`z` could be
        not on same device)

        :param n_batch: number of sentences to generate
        :param max_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param temp: temperature of softmax
        :return: list of tensors of strings, samples sequence x
        """
        def collate_fn(data):
            return list(filter(None.__ne__, data))

        if z is None:
            z = self.sample_z_prior(n_batch)
        else:
            n_batch = z.shape[0]

        z = z.to(self.device)
        # raw logits
        raw_logits = self.state_decoder(z, max_len).data.cpu().numpy()
        data = DataLoader(SDVAEGenerater(raw_logits, self.utils, use_random=True,
                                         sample_times=sample_times, valid=valid),
                          batch_size=n_batch,
                          num_workers=self.cmd_args.n_workers if
                          self.cmd_args.n_workers is not None else 0,
                          collate_fn=collate_fn,
                          worker_init_fn=set_torch_seed_to_all_gens
                          if self.cmd_args.n_workers is not None and
                             self.cmd_args.n_workers > 0 else None)
        return data.__iter__().__next__()

    def recon(self, x_inputs):
        z_mean, _ = self.encoder(x_inputs)

        return self.sample(z_mean.shape[0], z=z_mean, valid=False)

class MolVAE(nn.Module):
    def __init__(self, vocab, cmd_args):
        super(MolVAE, self).__init__()
        print('using vae')
        self.vocab = vocab
        self.latent_dim = cmd_args.d_z
        self.utils = MolUtil(cmd_args)
        self.encoder = get_encoder(cmd_args, self.utils)
        self.state_decoder = StateDecoder(max_len=cmd_args.max_decode_steps,
                                          latent_dim=cmd_args.d_z,
                                          DECISION_DIM=self.utils.DECISION_DIM)
        self.perp_calc = PerpCalculator()
        self.cmd_args = cmd_args
        self.onehot = True

    def reparameterize(self, mu, logvar):
        if self.training:
            eps = mu.data.new(mu.size()).normal_(0, self.cmd_args.eps_std)
            eps = eps.to(self.cmd_args.device)
            eps = Variable(eps)
            
            return mu + eps * torch.exp(logvar * 0.5)            
        else:
            return mu

    def forward(self, x_inputs, true_binary, rule_masks):        
        z_mean, z_log_var = self.encoder(x_inputs)

        z = self.reparameterize(z_mean, z_log_var)

        raw_logits = self.state_decoder(z)
        perplexity = self.perp_calc(true_binary, rule_masks, raw_logits)

        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), -1)
        
        return (perplexity, torch.mean(kl_loss))

    @property
    def device(self):
        return next(self.parameters()).device

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return torch.randn(n_batch, self.latent_dim,
                           device=self.device)

    def sample(self, n_batch, max_len=None, z=None, sample_times=100, valid=False, use_random=True):
        """Generating n_batch samples in eval mode (`z` could be
        not on same device)

        :param n_batch: number of sentences to generate
        :param max_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param temp: temperature of softmax
        :return: list of tensors of strings, samples sequence x
        """
        def collate_fn(data):
            return list(filter(None.__ne__, data))

        n_workers = self.cmd_args.n_workers if self.cmd_args.n_workers is not None else 0
        if z is None:
            z = self.sample_z_prior(n_batch)
        else:
            n_batch = max(1, z.shape[0] // max(1, n_workers))

        z = z.to(self.device)
        # raw logits
        raw_logits = self.state_decoder(z, max_len).permute(1, 0, 2).data.cpu().numpy()
        data = DataLoader(SDVAEGenerater(raw_logits, self.utils, use_random=use_random,
                                         sample_times=sample_times, valid=valid),
                          batch_size=n_batch,
                          num_workers=n_workers,
                          collate_fn=collate_fn, drop_last=True,
                          worker_init_fn=set_torch_seed_to_all_gens
                          if n_workers > 0 else None)
        return [_1  for _0 in data for _1 in _0]

    def recon(self, x_inputs, encode_times=10, decode_times=5, use_random=True):
        decode_results = []
        for _ in range(encode_times):
            z_mean, _ = self.encoder(x_inputs)
            if decode_times > 1:
                z_mean = z_mean.repeat(decode_times, 1)
            # for _ in range(decode_times):
            decode_results.append(np.array(self.sample(z_mean.shape[0], z=z_mean,
                                                       sample_times=1, use_random=use_random))
                                  .reshape((-1,decode_times), order='F'))
            del z_mean
            torch.cuda.empty_cache()
        return np.hstack(decode_results)
