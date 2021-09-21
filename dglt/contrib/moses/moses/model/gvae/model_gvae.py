from abc import ABC
from abc import abstractmethod
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.gumbel import Gumbel

from dglt.contrib.moses.moses.model.gvae.cvt_smiles_to_gidxs import Grammar
from dglt.contrib.moses.moses.model.gvae.utils import Flatten

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
        self.vocab_len = len(vocab) + 1
        self.ridx_pad = len(vocab)  # EOS-padding rule's index (the last one)

        # initialize the grammar model
        self.__init_grammar()

        # build encoder & decoder
        encoder = self.__build_encoder()
        decoder = self.__build_decoder()
        self.vae = nn.ModuleList([encoder, decoder])

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
        y = self.__decode(mu, logvar)

        # apply masks to decoded probabilistic outputs
        x_idxs = torch.cat([torch.unsqueeze(ix, dim=0) for ix in x], dim=0)
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

    def sample(self, n_batch, max_len=100):
        """Sample SMILES strings from the prior distribution.

        Args:
        * n_batch: # of SMILES strings
        * max_len: maximal length of a SMILES string

        Returns:
        * string_list: list of SMILES strings
        """

        with torch.no_grad():
            # use latent vector's prior distribution
            mu = torch.zeros(n_batch, self.config.d_z).to(self.device)
            logvar = torch.zeros(n_batch, self.config.d_z).to(self.device)

            # decoding
            y = self.__decode(mu, logvar)

            # convert into SMILES strings
            probs_list = [torch.squeeze(v) for v in torch.split(y, 1, dim=0)]
            smiles_list = [self.probs2smiles(v) for v in probs_list]

        print('reconstructed: ' + smiles_list[0])

        return smiles_list

    def recon(self, x_idxs_list, max_len=100):
        """Reconstruct SMILES strings from list of one-hot entries' indices.

        Args:
        * x_idxs_list: list of one-hot entries' indices
        * max_len: maximal length of a SMILES string

        Returns:
        * y: reconstructed SMILES strings
        """

        with torch.no_grad():
            # pre-processing
            x = tuple(self.raw2tensor(x) for x in x_idxs_list)

            # encoding & decoding
            mu, logvar = self.__encode(x)
            y = self.__decode(mu, logvar)

            # convert into SMILES strings
            probs_list = [torch.squeeze(v) for v in torch.split(y, 1, dim=0)]
            smiles_list = [self.probs2smiles(v) for v in probs_list]

        print('original:      ' + self.grammar.to_smiles(x_idxs_list[0]))
        print('reconstructed: ' + smiles_list[0])

        return smiles_list

    def raw2tensor(self, raw_data, device='model'):
        """Convert the raw data to torch.tensor.

        Args:
        * raw_data: '|'-separated non-zero entries' indices string
        * device: where to place the torch.tensor

        Returns:
        * tensor: torch.tensor consists of one-hot entries' indices
        """

        ids = [int(sub_str) for sub_str in raw_data.split('|')]
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
        # configure parameters for convolutional and linear layers
        conv1_param = (self.vocab_len, 9, 9)  # in_channels / out_channels / kernel_size
        conv2_param = (9, 9, 9)
        conv3_param = (9, 10, 11)
        nb_idims_fc = conv3_param[1] * (self.config.smiles_maxlen -
            (conv1_param[2] - 1) - (conv2_param[2] - 1) - (conv3_param[2] - 1))
        nb_odims_fc = 435

        # encoder sub-network
        self.encoder_rnn = nn.Sequential(
            nn.Conv1d(conv1_param[0], conv1_param[1], conv1_param[2]),
            nn.ReLU(),
            nn.Conv1d(conv2_param[0], conv2_param[1], conv2_param[2]),
            nn.ReLU(),
            nn.Conv1d(conv3_param[0], conv3_param[1], conv3_param[2]),
            nn.ReLU(),
            Flatten(),
            nn.Linear(nb_idims_fc, nb_odims_fc),
            nn.ReLU()
        )

        # latent vector's mean & variance
        self.q_mu = nn.Linear(nb_odims_fc, self.config.d_z)
        self.q_logvar = nn.Linear(nb_odims_fc, self.config.d_z)

        # packe all modules into one
        encoder = nn.ModuleList([
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
        ])

        return encoder

    def __build_decoder(self):
        # map the latent vector for decoding
        self.decoder_map = nn.Sequential(
            nn.Linear(self.config.d_z, self.config.d_z),
            nn.ReLU()
        )

        # VAE's latent vector -> GRU's latent vector
        self.decoder_lat = nn.Linear(self.config.d_z, self.config.d_d_h)

        # decoder sub-network
        self.decoder_rnn = nn.GRU(
            self.config.d_z,
            self.config.d_d_h,
            num_layers=self.config.d_n_layers,
            batch_first=True,
            dropout=self.config.d_dropout if self.config.d_n_layers > 1 else 0
        )

        # probabilistic outputs
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.config.d_d_h, self.vocab_len),
            nn.Softplus()
        )

        # pack all modules into one
        decoder = nn.ModuleList([
            self.decoder_map,
            self.decoder_lat,
            self.decoder_rnn,
            self.decoder_fc
        ])

        return decoder

    def __encode(self, x):
        x = torch.cat([torch.unsqueeze(v, dim=0) for v in x], dim=0)
        x_expd = torch.zeros(x.size()[0], self.vocab_len, x.size()[1], device=x.device)
        x_expd.scatter_(1, x.view([x.size()[0], 1, x.size()[1]]), 1)
        x_encoded = self.encoder_rnn(x_expd)
        mu = self.q_mu(x_encoded)
        logvar = self.q_logvar(x_encoded)

        return mu, logvar

    def __decode(self, mu, logvar):
        z = mu + (logvar / 2).exp() * torch.randn_like(mu)
        z_mapped = self.decoder_map(z)
        z_tiled = z_mapped.unsqueeze(1).repeat(1, self.config.smiles_maxlen, 1)
        h = self.decoder_lat(z)
        h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
        z_decoded, __ = self.decoder_rnn(z_tiled, h)
        y = self.decoder_fc(z_decoded)

        return y

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
        x_prob_msk = mask * x_prob.view([-1, self.vocab_len])
        x_prob_fnl = x_prob_msk.view(x_prob.size())

        # apply the mask (if <x_prob> is not all positive)
        #x_prob_min = torch.min(x_prob) - 1.0  # ensure non-zero gradients
        #x_prob_min = x_prob_min.detach()  # stop gradients
        #x_prob_msk = mask * (x_prob - x_prob_min).view([-1, self.vocab_len]) + x_prob_min
        #x_prob_fnl = x_prob_msk.view(x_prob.size())

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
