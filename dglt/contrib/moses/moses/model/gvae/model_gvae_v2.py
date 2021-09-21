from abc import ABC
from abc import abstractmethod
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.gumbel import Gumbel
from torch.distributions.normal import Normal

from dglt.contrib.moses.moses.model.gvae.cvt_smiles_to_gidxs import Grammar
from dglt.contrib.moses.moses.model.gvae.utils import Flatten
from dglt.contrib.moses.moses.model.gvae.model_props import Model as PropsModel
from dglt.contrib.moses.moses.model.gvae.model_props import FeatOptimizer

class Model(nn.Module):
    """Grammar Variational Autoencoder - 1st & 2nd version."""

    def __init__(self, vocab, config):
        """Constructor function.

        Args:
        * vocab: model's vocabulary
        * config: model's configuration
        """

        super().__init__()

        # get the vocabulary and configurations
        self.vocab = vocab
        self.config = config
        self.vocab_len = len(vocab) + 2 # BOS & EOS padding rules
        self.ridx_bos = len(vocab) + 0  # BOS-padding rule's index (exactly one)
        self.ridx_pad = len(vocab) + 1  # EOS-padding rule's index (zero, one, or more)

        # initialize the grammar model
        self.__init_grammar()

        # build encoder & decoder
        encoder = self.__build_encoder()
        decoder = self.__build_decoder()
        self.vae = nn.ModuleList([encoder, decoder])

        # initialize the feature optimizer
        self.feat_optimizer = None

    def forward(self, x):
        """Perform the forward passing and compute losses.

        Args:
        * x: list of torch.tensors, one per training sample

        Returns:
        * kl_loss: KL-divergence loss
        * recon_loss: reconstruction loss
        """

        # encoding & decoding
        mu, logvar = self.__encode(x)
        y = self.__decode(x, mu, logvar)
        y = y[:, :-1, :].contiguous()  # remove the last prediction

        # apply masks to decoded probabilistic outputs
        x_idxs = torch.cat([torch.unsqueeze(ix, dim=0) for ix in x], dim=0)
        x_idxs = x_idxs[:, 1:].contiguous()  # remove the BOS-padding rule
        y_mask = self.__mask_decoded_smiles(x_idxs, y)

        # compute KL-divergence & re-construction losses
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        recon_loss = F.cross_entropy(
            y_mask.view([-1, self.vocab_len]), x_idxs.view([-1]), ignore_index=self.ridx_pad)
        recon_loss *= self.config.smiles_maxlen

        # additional metrics
        y_idxs = torch.argmax(y_mask.view([-1, self.vocab_len]), dim=-1)
        x_idxs = x_idxs.view([-1])
        acc_all = torch.mean((y_idxs == x_idxs).type(torch.FloatTensor))
        cnt_true = torch.sum((y_idxs == x_idxs).type(torch.FloatTensor) * (x_idxs != self.ridx_pad).type(torch.FloatTensor))
        cnt_full = torch.sum((x_idxs != self.ridx_pad).type(torch.FloatTensor))
        acc_vld = cnt_true / cnt_full
        cnt_true = torch.sum((y_idxs == x_idxs).type(torch.FloatTensor) * (x_idxs == self.ridx_pad).type(torch.FloatTensor))
        cnt_full = torch.sum((x_idxs == self.ridx_pad).type(torch.FloatTensor))
        acc_pad = cnt_true / cnt_full

        return kl_loss, recon_loss, acc_all, acc_vld, acc_pad

    def sample(self, nb_smpls, max_len=100):
        """Sample SMILES strings from the prior distribution.

        Args:
        * nb_smpls: # of SMILES strings
        * max_len: maximal length of a SMILES string

        Returns:
        * string_list: list of SMILES strings
        """

        smiles_list = []
        with torch.no_grad():
            mu = torch.zeros(nb_smpls, self.config.d_z).to(self.device)
            logvar = torch.ones(nb_smpls, self.config.d_z).to(self.device)
            mu_list = torch.split(mu, 1, dim=0)
            logvar_list = torch.split(logvar, 1, dim=0)
            for mu_sel, logvar_sel in zip(mu_list, logvar_list):
                idxs = self.__sample_rule_idxs(mu_sel, logvar_sel)
                idxs_str = '|'.join([str(idx) for idx in idxs])
                smiles = self.grammar.to_smiles(idxs_str)
                smiles_list.append(smiles)

        return smiles_list

    def recon(self, x_idxs_list, max_len=100):
        """Reconstruct SMILES strings from list of one-hot entries' indices.

        Args:
        * x_idxs_list: list of one-hot entries' indices
        * max_len: maximal length of a SMILES string

        Returns:
        * y: reconstructed SMILES strings
        """

        smiles_list = []
        with torch.no_grad():
            x = tuple(self.raw2tensor(x) for x in x_idxs_list)
            mu, logvar = self.__encode(x)
            mu_list = torch.split(mu, 1, dim=0)
            logvar_list = torch.split(logvar, 1, dim=0)
            for mu_sel, logvar_sel in zip(mu_list, logvar_list):
                idxs = self.__sample_rule_idxs(mu_sel, logvar_sel)
                idxs_str = '|'.join([str(idx) for idx in idxs])
                smiles = self.grammar.to_smiles(idxs_str)
                smiles_list.append(smiles)

        return smiles_list

    def design(self, nb_smpls, properties, max_len=100):
        """Design SMILES strings that satisfy given property values.

        Args:
        * nb_smpl: # of SMILES strings
        * properties: dict of desired property values
        * max_len: maximal length of a SMILES string

        Returns:
        * smiles_list: designed SMILES strings
        """

        # initialize the feature optimizer at the first entry
        if self.feat_optimizer is None:
            self.__init_feat_optimizer()

        # optimize latent vector's statistics to satisfy given property values
        feats_init = Normal(0.0, 1.0).sample((nb_smpls, self.config.d_z)) * 1e-2
        #labl_values = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9]]).repeat(nb_smpls, 1)
        #labl_values = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1]]).repeat(nb_smpls, 1)
        #labl_values = torch.tensor([[0.6869, 0.1786, 0.2979, 0.4056, 0.2218]]).repeat(nb_smpls, 1)
        #labl_values = torch.tensor([[0.3517, 0.6880, 0.4520, 0.5719, 0.2058]]).repeat(nb_smpls, 1)
        labl_values = torch.tensor([[0.6588, 0.2467, 0.3372, 0.7682, 0.5377]]).repeat(nb_smpls, 1)
        labl_coeffs = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]).repeat(nb_smpls, 1)
        feats_finl, preds_finl = self.feat_optimizer.run(feats_init, labl_values, labl_coeffs)

        # decode SMILES strings from optimized latent vector's statistics
        smiles_list = []
        with torch.no_grad():
            mu = feats_finl.to(self.device)
            logvar = -float('inf') * torch.ones(nb_smpls, self.config.d_z).to(self.device)
            mu_list = torch.split(mu, 1, dim=0)
            logvar_list = torch.split(logvar, 1, dim=0)
            for mu_sel, logvar_sel in zip(mu_list, logvar_list):
                idxs = self.__sample_rule_idxs(mu_sel, logvar_sel)
                idxs_str = '|'.join([str(idx) for idx in idxs])
                smiles = self.grammar.to_smiles(idxs_str)
                smiles_list.append(smiles)

        return smiles_list

    def __init_feat_optimizer(self):
        """Initialize the feature optimizer."""

        # restore a pre-trained model for feature optimization
        model = PropsModel()
        model_state = torch.load('/data1/jonathan/Molecule.Generation/AIPharmacist-models/gvae_props.model')
        model.load_state_dict(model_state)
        self.feat_optimizer = FeatOptimizer(model)

    def raw2tensor(self, raw_data, device='model'):
        """Convert the raw data to torch.tensor.

        Args:
        * raw_data: '|'-separated non-zero entries' indices string
        * device: where to place the torch.tensor

        Returns:
        * tensor: torch.tensor consists of one-hot entries' indices
        """

        ids = [self.ridx_bos]
        ids += [int(sub_str) for sub_str in raw_data.split('|')]
        ids += [self.ridx_pad] * (self.config.smiles_maxlen - len(ids))
        tensor = torch.tensor(ids, device=self.device if device == 'model' else device)

        return tensor

    def probs2smiles(self, probs):
        """Convert decoded probabilistic outputs to SMILES string.

        Args:
        * probs: decoded probabilistic outputs

        Returns:
        * smiles: SMILES string
        """

        idxs = self.__sample_onehot_indices(probs)
        idxs_str = '|'.join([str(idx) for idx in idxs])
        smiles = self.grammar.to_smiles(idxs_str)

        return smiles

    @property
    def device(self):
        """The model's device."""

        return next(self.parameters()).device

    @property
    def optim_params(self):
        """Get model parameters to be optimized."""

        return (p for p in self.parameters() if p.requires_grad)

    def __init_grammar(self):
        """Initialize the grammar model."""

        # initialize the grammar model
        self.grammar = Grammar(self.config.rule_path)

        # obtain a list of grammar rules and LHS symbols
        self.lhs_uniq = []
        lhs_full = [rule.split('->')[0].strip() for rule in self.vocab]
        for lhs in lhs_full:
            if lhs not in self.lhs_uniq:
                self.lhs_uniq.append(lhs)
        self.lhs_init = self.lhs_uniq[0]

        # create masks for LHS-rule mapping
        self.mask_l2r = torch.zeros(len(self.lhs_uniq), self.vocab_len)
        for rule_idx, rule in enumerate(self.vocab):
            lhs_idx = self.lhs_uniq.index(rule.split('->')[0].strip())
            self.mask_l2r[lhs_idx, rule_idx] = 1.0

        # create masks for rule-rule mapping
        self.mask_r2r = torch.zeros(self.vocab_len, self.vocab_len)
        self.mask_r2r[-1, -1] = 1.0
        for rule_idx, rule in enumerate(self.vocab):
            lhs_idx = self.lhs_uniq.index(rule.split('->')[0].strip())
            self.mask_r2r[rule_idx] = self.mask_l2r[lhs_idx]

    def __build_encoder(self):
        # word embedding layer
        self.x_emb = nn.Embedding(self.vocab_len, self.config.d_emb, self.ridx_pad)
        if self.config.freeze_embeddings:
            self.x_emb.weight.requires_grad = False

        # encoder
        self.encoder_rnn = nn.GRU(
            self.config.d_emb,
            self.config.q_d_h,
            num_layers=self.config.q_n_layers,
            batch_first=True,
            dropout=self.config.q_dropout if self.config.q_n_layers > 1 else 0,
            bidirectional=self.config.q_bidir
        )

        # latent vector's mean & variance
        q_d_last = self.config.q_d_h * (2 if self.config.q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last, self.config.d_z)
        self.q_logvar = nn.Linear(q_d_last, self.config.d_z)

        # packe all modules into one
        encoder = nn.ModuleList([
            self.x_emb,
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
        ])

        return encoder

    def __build_decoder(self):
        # VAE's latent vector -> GRU's latent vector
        self.decoder_lat = nn.Linear(self.config.d_z, self.config.d_d_h)

        # decoder sub-network
        self.decoder_rnn = nn.GRU(
            self.config.d_emb + self.config.d_z,
            self.config.d_d_h,
            num_layers=self.config.d_n_layers,
            batch_first=True,
            dropout=self.config.d_dropout if self.config.d_n_layers > 1 else 0
        )

        # probabilistic outputs
        self.decoder_fc = nn.Linear(self.config.d_d_h, self.vocab_len)

        # pack all modules into one
        decoder = nn.ModuleList([
            self.decoder_lat,
            self.decoder_rnn,
            self.decoder_fc
        ])

        return decoder

    def encode(self, x):
        return self.__encode(x)

    def __encode(self, x):
        x = [self.x_emb(v) for v in x]
        x = nn.utils.rnn.pack_sequence(x)
        __, h = self.encoder_rnn(x, None)
        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)

        return mu, logvar

    def __decode(self, x, mu, logvar):
        lengths = [len(v) for v in x]
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.ridx_pad)
        x_emb = self.x_emb(x)
        z = mu + (logvar / 2).exp() * torch.randn_like(mu)
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths, batch_first=True)
        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
        output, __ = self.decoder_rnn(x_input, h_0)
        output, __ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)

        return y

    def __sample_rule_idxs(self, mu, logvar):
        """Sample indices of grammar rules.

        Note: both <mu> and <logvar> correspond to a single instance, instead of a mini-batch.
        """

        # hyper-parameters
        ridxs_maxlen = 300  # maximal length of rule indices
        softmax_temp = 0.5  # softmax function's temperature

        # send <mask_r2r> to the model's device
        if self.mask_l2r.device != self.device:
            self.mask_l2r = self.mask_l2r.to(self.device)

        # compute the latent vector and initialize the decoder's hidden state
        z = mu + (logvar / 2).exp() * torch.randn_like(mu)
        z_0 = z.unsqueeze(1)  # no need to repeat
        h = self.decoder_lat(z)
        h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

        # sample rules in a step-by-step manner
        rule_idxs = []
        stack = [self.lhs_init]
        x_last = self.ridx_bos * torch.ones(1, dtype=torch.long, device=self.device)
        while stack and len(rule_idxs) < ridxs_maxlen:
            # pop the top element from stack
            lhs, stack = stack[-1], stack[:-1]
            lhs_idx = self.lhs_uniq.index(lhs)

            # generate probabilistic outputs with mask applied
            x_emb = self.x_emb(x_last).unsqueeze(1)
            x_cmb = torch.cat([x_emb, z_0], dim=-1)
            outputs, h = self.decoder_rnn(x_cmb, h)
            y = self.decoder_fc(outputs.squeeze(1))
            y_min = torch.min(y) - 1.0
            y = y_min + (y - y_min) * self.mask_l2r[lhs_idx].view([1, -1])
            y = F.softmax(y / softmax_temp, dim=-1) * self.mask_l2r[lhs_idx].view([1, -1])

            # sample a one-hot entries' index
            x_last = torch.multinomial(y, 1)[:, 0]
            rule_idx = x_last.item()
            rule_idxs.append(rule_idx)

            # push non-terminal RHS symbols into stack
            rhs_list = self.vocab[rule_idx].split('->')[1].strip().split()
            for rhs in rhs_list[::-1]:
                if not (rhs.startswith('\'') and rhs.endswith('\'')):
                    stack.append(rhs)

        return rule_idxs

    def __mask_decoded_smiles(self, x_idxs, x_prob):
        """Apply masks to decoded probabilistic outputs.

        Args:
        * x_idxs: ground-truth one-hot entries' indices
        * x_prob: decoded probabilistic outputs

        Returns:
        * x_prob_fnl: decoded probabilistic outputs, with masks applied
        """

        # send <mask_r2r> to the model's device
        if self.mask_r2r.device != self.device:
            self.mask_r2r = self.mask_r2r.to(self.device)

        # extract the mask matrix
        mask = torch.index_select(self.mask_r2r, 0, x_idxs.view([-1]))

        # apply the mask (if <x_prob> is all positive)
        #x_prob_msk = mask * x_prob.view([-1, self.vocab_len])
        #x_prob_fnl = x_prob_msk.view(x_prob.size())

        # apply the mask (if <x_prob> is not all positive)
        x_prob_min = torch.min(x_prob) - 1.0  # ensure non-zero gradients
        x_prob_min = x_prob_min.detach()  # stop gradients
        x_prob_msk = mask * (x_prob - x_prob_min).view([-1, self.vocab_len]) + x_prob_min
        x_prob_fnl = x_prob_msk.view(x_prob.size())

        return x_prob_fnl

    def __sample_onehot_indices(self, x_dec):
        """Sample one-hot entries' indices from probabilistic outputs, with masks applied.

        Args:
        * x_dec: decoded probabilistic outputs

        Returns:
        * x_idxs: decoded one-hot entries' indices
        """

        eps = 1e-8

        # send <mask_l2r> to the model's device
        if self.mask_l2r.device != self.device:
            self.mask_l2r = self.mask_l2r.to(self.device)

        # sample one-hot entries' indices from probabilistic outputs
        step = 0  # timestamp
        stack = [self.lhs_init]
        x_idxs = []
        m = Gumbel(torch.tensor([0.0]), torch.tensor([0.1]))
        while stack and step < x_dec.size()[0]:
            # pop the top element from stack
            lhs = stack[-1]
            stack = stack[:-1]
            lhs_idx = self.lhs_uniq.index(lhs)

            # sample a rule from masked probabilistic outputs
            probs = (x_dec[step].exp() * self.mask_l2r[lhs_idx] + eps).log()
            probs += torch.squeeze(m.sample(probs.size())).to(self.device)
            rule_idx = torch.argmax(probs)
            x_idxs.append(rule_idx.item())

            '''
            # sample a rule from masked probabilistic outputs
            probs = F.softmax(x_dec[step], dim=0)
            probs += torch.squeeze(m.sample(probs.size())).to(self.device) * 0
            probs_min = torch.min(probs) - 1.0
            probs = self.mask_l2r[lhs_idx] * (probs - probs_min) + probs_min
            rule_idx = torch.argmax(probs)
            x_idxs.append(rule_idx.item())
            '''

            '''
            # sample a rule from masked probabilistic outputs
            probs = x_dec[step]
            probs_min = torch.min(probs) - 1.0
            probs = self.mask_l2r[lhs_idx] * (probs - probs_min) + probs_min
            probs = F.softmax(probs, dim=0)
            probs *= self.mask_l2r[lhs_idx]
            probs /= probs.sum()
            rule_idx = torch.multinomial(probs, 1)
            x_idxs.append(rule_idx.item())
            '''

            # push non-terminal RHS symbols into stack
            rhs_list = self.vocab[rule_idx].split('->')[1].strip().split()
            for rhs in rhs_list[::-1]:
                if not (rhs.startswith('\'') and rhs.endswith('\'')):
                    stack.append(rhs)

            # increase <step> by one
            step += 1

        return x_idxs
