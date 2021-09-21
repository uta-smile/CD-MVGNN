from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dglt.contrib.moses.moses.model.sd_vae.model.attribute_tree_vae_decoder import AttMolGraphDecoders
from dglt.contrib.moses.moses.model.sd_vae.model.mol_decoder import PerpCalculator
from dglt.contrib.moses.moses.model.sd_vae.utils.mol_util import MolUtil

from dglt.contrib.moses.moses.model.sd_vae.utils.mol_tree import get_smiles_from_tree


class VAE(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()

        self.n_workers = config.n_workers if config.n_workers is not None else 0
        # self.vocabulary = vocab
        self.utils = MolUtil(config)

        # spectial elements
        self.pad = self.utils.DECISION_DIM - 1
        self.bos = 0
        self.onehot = False

        # Word embeddings layer
        n_vocab, d_emb = self.utils.DECISION_DIM, self.utils.DECISION_DIM
        self.n_vocab = n_vocab
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.utils.DECISION_DIM - 1)
        self.x_emb.weight.data.copy_(torch.eye(self.utils.DECISION_DIM))
        if config.freeze_embeddings:
            self.x_emb.weight.requires_grad = False

        self.z_emb = nn.Embedding(n_vocab, d_emb, self.utils.DECISION_DIM - 1)
        self.z_emb.weight.data.copy_(torch.eye(self.utils.DECISION_DIM))
        self.z_emb.weight.requires_grad = False

        # Encoder
        if config.rnn_type == 'gru':
            self.encoder_rnn = nn.GRU(
                d_emb,
                config.q_d_h,
                num_layers=config.q_n_layers,
                batch_first=True,
                dropout=config.q_dropout if config.q_n_layers > 1 else 0,
                bidirectional=config.q_bidir
            )
        else:
            raise ValueError(
                "Invalid q_cell type, should be one of the ('gru',)"
            )

        q_d_last = config.q_d_h * (2 if config.q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last, config.d_z)
        self.q_logvar = nn.Linear(q_d_last, config.d_z)

        # Decoder
        if config.rnn_type == 'gru':
            self.decoder_rnn = nn.GRU(
                d_emb + config.d_z,
                config.d_d_h,
                num_layers=config.d_n_layers,
                batch_first=True,
                dropout=config.d_dropout if config.d_n_layers > 1 else 0
            )
        else:
            raise ValueError(
                "Invalid d_cell type, should be one of the ('gru',)"
            )

        self.decoder_lat = nn.Linear(config.d_z, config.d_d_h)
        self.decoder_fc = nn.Linear(config.d_d_h, n_vocab)

        # Loss
        self.recon_loss = PerpCalculator()

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
        ])
        self.decoder = nn.ModuleList([
            self.decoder_rnn,
            self.decoder_lat,
            self.decoder_fc
        ])
        self.vae = nn.ModuleList([
            self.x_emb,
            self.encoder,
            self.decoder,
            self.recon_loss
        ])

    @property
    def device(self):
        return next(self.parameters()).device

    def string2tensor(self, string, device='model'):
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(
            ids, dtype=torch.long,
            device=self.device if device == 'model' else device
        )

        return tensor

    def forward(self, x, masks):
        """Do the VAE forward step

        :param x: list of tensors of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        # Encoder: x -> z, kl_loss
        z, kl_loss, _ = self.forward_encoder(x)

        # Decoder: x, z -> recon_loss
        recon_loss = self.forward_decoder(x, z, masks)

        return recon_loss, kl_loss

    def forward_encoder(self, x):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)

        _, h = self.encoder_rnn(x, None)

        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)

        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar.clamp(-1e30, 10) / 2).exp() * eps

        kl_loss = 0.5 * (logvar.clamp(-1e30, 10).exp() + mu ** 2 - 1 - logvar).sum(1).mean()

        return z, kl_loss, mu

    def forward_decoder(self, x, z, masks):
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        """

        lengths = [len(i_x) for i_x in x]

        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.pad)
        x_emb = self.x_emb(x)
        true_binary = self.z_emb(x)

        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True)

        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

        output, _ = self.decoder_rnn(x_input, h_0)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)

        recon_loss = self.recon_loss(true_binary[:,1:], masks[:,1:y.shape[1]], y[:, :-1])
        # recon_loss = F.cross_entropy(
        #     y[:, :-1].contiguous().view(-1, y.size(-1)),
        #     x[:, 1:].contiguous().view(-1),
        #     ignore_index=self.pad
        # )

        return recon_loss

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return torch.randn(n_batch, self.q_mu.out_features,
                           device=self.x_emb.weight.device)

    def sample(self, n_batch, max_len, z=None, temp=1.0,
               sample_times=10, valid=False, use_random=True):
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
        z_0 = z.unsqueeze(1)

        data = []
        for _ in range(sample_times):
            # Initial values
            h = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)

            # Generating cycle
            dec = AttMolGraphDecoders(n_batch, self.utils, use_random=use_random)
            dec.getCandidate(np.zeros((n_batch, self.n_vocab)), 0, self.pad)
            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder_rnn(x_input, h)
                raw_y = self.decoder_fc(o.squeeze(1)) / temp
                y = F.softmax(raw_y, dim=-1)

                w = torch.tensor(dec.getCandidate(raw_y.data.cpu().numpy(), i, self.pad),
                                     device=self.device, dtype=torch.long)
                # raw_logits.append(y)

                if np.all(dec.eos_mark):
                    break

            # raw_logits = torch.stack(raw_logits, 1).data.cpu().numpy()

            # Converting `raw_logits` to list of tensors
            results = []
            for node, eos, junk in zip(dec.nodes, dec.eos_mark, dec.junks):
                import random, string
                try:
                    if eos and not junk:
                        results.append(get_smiles_from_tree(node))
                    elif junk:
                        results.append('JUNK' + ''.join(
                            random.choice(string.ascii_uppercase + string.digits) for _ in range(256)))
                    else:
                        results.append('JUNK-EX' + ''.join(
                            random.choice(string.ascii_uppercase + string.digits) for _ in range(256)))
                except:
                    results.append('JUNK-ERR' + ''.join(
                        random.choice(string.ascii_uppercase + string.digits) for _ in range(256)))
            data.append(results)

        decodes = []
        for result in zip(*data):
            cnt = Counter()
            for res in result:
                if not res.startswith('JUNK'):
                    cnt[res] += 1
            # print(cnt)
            if len(cnt) > 0:
                decodes.append(cnt.most_common(1)[0][0])
            else:
                if valid:
                    continue
                decodes.append(result[0])

        return decodes
        # data = DataLoader(SDVAEGenerater(raw_logits, self.utils, use_random=use_random,
        #                                  sample_times=sample_times, valid=valid),
        #                   batch_size=n_batch,
        #                   num_workers=self.n_workers,
        #                   collate_fn=collate_fn, drop_last=True,
        #                   worker_init_fn=set_torch_seed_to_all_gens
        #                   if self.n_workers > 0 else None)
        # return [_1 for _0 in data for _1 in _0]

    def recon(self, x_inputs, max_len, encode_times=10, decode_times=5, use_random=True):
        decode_results = []
        for _ in range(encode_times):
            _, _, z_mean = self.forward_encoder(x_inputs)
            if decode_times > 1:
                z_mean = z_mean.repeat(decode_times, 1)
            # for _ in range(decode_times):
            decode_results.append(np.array(self.sample(z_mean.shape[0], max_len, z=z_mean,
                                                       sample_times=1, use_random=use_random))
                                  .reshape((-1,decode_times), order='F'))
            del z_mean
            torch.cuda.empty_cache()
        return np.hstack(decode_results)
