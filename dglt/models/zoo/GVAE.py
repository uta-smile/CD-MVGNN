import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.gumbel import Gumbel
from torch.distributions.normal import Normal

# TODO: move Grammar to models?
from dglt.contrib.moses.moses.model.gvae.cvt_smiles_to_gidxs import Grammar
from ..modules import GRUDecoder, GRUEncoder
from ..layers import GaussianSampling


class GrammarDecoder(GRUDecoder):
    def sample(self, n_batch=1, z=None, max_len=100, temp=1.0, device=None):
        if device is None:
            device = self.decoder_lat.weight.device

        smiles_list = []
        with torch.no_grad:
            if z is None:
                z = torch.randn(n_batch, self.d_z, device=device)
            else:
                n_batch = z.shape[0]

            z_list = torch.split(z, 1, dim=0)
            for z_sel in z_list:
                idxs = self.__sample_rule_idxs(z_sel)
                idxs_str = '|'.join([str(idx) for idx in idxs])
                smiles = self.grammar.to_smiles(idxs_str)
                smiles_list.append(smiles)

        return smiles_list

    def __sample_rule_idxs(self, z):
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


class GVAE(nn.Module):
    """GVAE model"""
    def __init__(self,
                 vocab,
                 encoder,
                 decoder,
                 sampling=None,
                 freeze_embedding=False):
        """
        Initializes the GVAE

        :param vocab: vocabulary for GVAE
        :param encoder: the encoder module
        :param decoder: the decoder module
        :param sampling: the sampling layer. AutoEncoder if None
        :param freeze_embeddings: if True freeze the weights of embeddings
        """
        super(GVAE, self).__init__()

        # get the vocabulary and configurations
        self.vocab = vocab
        self.vocab_len = len(vocab) + 2 # BOS & EOS padding rules
        self.ridx_bos = len(vocab) + 0  # BOS-padding rule's index (exactly one)
        self.ridx_pad = len(vocab) + 1  # EOS-padding rule's index (zero, one, or more)

        # initialize the grammar model
        self.__init_grammar()

        # word embedding layer
        self.x_emb = nn.Embedding(self.vocab_len, self.config.d_emb, self.ridx_pad)
        if self.config.freeze_embeddings:
            self.x_emb.weight.requires_grad = False

        # build encoder & decoder
        self.encoder = encoder
        self.sampling = sampling
        if self.sampling is None:
            self.encoder_sample = self.encoder
        else:
            self.encoder_sample = torch.nn.Sequential(
                encoder,
                sampling
            )

        self.decoder = decoder
        self.vae = nn.ModuleList([
            self.x_emb,
            self.encoder_sample,
            self.decoder
        ])

    def forward(self, x):
        x_encoder = [self.x_emb(i_x) for i_x in x]
        x_encoder = nn.utils.rnn.pack_sequence(x_encoder)

        lengths = [len(i_x) for i_x in x]
        x_decoder = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                              padding_value=self.pad)
        x_decoder = self.x_emb(x_decoder)

        if self.sampling is None:
            z = self.encoder_sample(x_encoder)
            mu = None
            logvar = None
        else:
            z, mu, logvar = self.encoder_sample(x_encoder)

        output = self.decoder(z, x_decoder, lengths)
        return output, mu, logvar

    def get_loss_function(self):
        # send <mask_r2r> to the model's device
        if self.mask_r2r.device != self.device:
            self.mask_r2r = self.mask_r2r.to(self.device)

        def loss_func(input, target, mu=None, logvar=None, kl_weight=1.0, mask_r2r=self.mask_r2r):
            input = input[:, :-1, :].contiguous()
            # apply masks to decoded probabilistic outputs
            x_idxs = torch.cat([torch.unsqueeze(ix, dim=0) for ix in target], dim=0)
            x_idxs = x_idxs[:, 1:].contiguous()  # remove the BOS-padding rule

            # apply masks for smiles
            mask = torch.index_select(mask_r2r, 0, x_idxs.view([-1]))

            # apply the mask (if <x_prob> is not all positive)
            x_prob_min = torch.min(input) - 1.0  # ensure non-zero gradients
            x_prob_min = x_prob_min.detach()  # stop gradients
            x_prob_msk = mask * (input - x_prob_min).view([-1, self.vocab_len]) + x_prob_min
            input_mask = x_prob_msk.view(input.size())

            # compute KL-divergence & re-construction losses
            if mu is not None and logvar is not None:
                kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
            else:
                kl_loss = torch.tensor(0.0, device=self.device)
            recon_loss = F.cross_entropy(
                input_mask.view([-1, self.vocab_len]), x_idxs.view([-1]), ignore_index=self.ridx_pad)
            recon_loss *= self.config.smiles_maxlen

            # additional metrics
            y_idxs = torch.argmax(input_mask.view([-1, self.vocab_len]), dim=-1)
            x_idxs = x_idxs.view([-1])
            acc_all = torch.mean((y_idxs == x_idxs).type(torch.FloatTensor))
            cnt_true = torch.sum(
                (y_idxs == x_idxs).type(torch.FloatTensor) * (x_idxs != self.ridx_pad).type(
                    torch.FloatTensor))
            cnt_full = torch.sum((x_idxs != self.ridx_pad).type(torch.FloatTensor))
            acc_vld = cnt_true / cnt_full
            cnt_true = torch.sum(
                (y_idxs == x_idxs).type(torch.FloatTensor) * (x_idxs == self.ridx_pad).type(
                    torch.FloatTensor))
            cnt_full = torch.sum((x_idxs == self.ridx_pad).type(torch.FloatTensor))
            acc_pad = cnt_true / cnt_full

            return kl_weight * kl_loss + recon_loss, {'kl_loss': kl_loss,
                                                       'recon_loss': recon_loss,
                                                       'acc_all': acc_all,
                                                       'acc_vld': acc_vld,
                                                       'acc_pad': acc_pad}
        return loss_func

    def sample(self, nb_smpls):
        return self.decoder.sample(n_batch=nb_smpls)

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

    @property
    def device(self):
        """The model's device."""

        return next(self.parameters()).device


class moseGVAE(GVAE):
    def __init__(self, vocab, config):
        n_vocab, d_emb = len(vocab), vocab.vectors.size(1)
        encoder = GRUEncoder(d_emb, config.q_d_h,
                             num_layers=config.q_n_layers,
                             dropout=config.q_dropout,
                             bidirectional=config.q_bidir)
        decoder = GrammarDecoder(d_emb, config.d_z, config.d_d_h, n_vocab,
                                 num_layers=config.d_n_layers,
                                 dropout=config.d_dropout)
        sample = GaussianSampling(config.q_d_h, config.d_z, bidirectional=config.q_bidir)
        super(moseGVAE, self).__init__(vocab, encoder, decoder, sampling=sample)

    def forward(self, x):
        output, mu, logvar = super(moseGVAE, self).forward(x)

        loss_func = self.get_loss_function()
        _, loss_metric = loss_func(output, x, mu=mu, logvar=logvar)
        return loss_metric['kl_loss'], loss_metric['recon_loss'], \
               loss_metric['acc_all'], loss_metric['acc_vld'], loss_metric['acc_pad']

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
    def optim_params(self):
        """Get model parameters to be optimized."""

        return (p for p in self.parameters() if p.requires_grad)

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

