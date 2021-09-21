import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter as P


class VAE(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()

        self.vocabulary = vocab
        self.design_col_names = config.design_col_names
        self.y_dim = len(config.design_col_names)
        self.d_z = config.d_z
        self.load_norm = False
        if self.y_dim > 0:
            self.min = P(torch.zeros(self.y_dim), requires_grad=False)
            self.max = P(torch.ones(self.y_dim), requires_grad=False)
        else:
            self.min = P(torch.tensor(0.), requires_grad=False)
            self.max = P(torch.tensor(1.), requires_grad=False)

        # Special symbols
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(vocab, ss))

        # Word embeddings layer
        n_vocab, d_emb = len(vocab), vocab.vectors.size(1)
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.pad)
        self.x_emb.weight.data.copy_(vocab.vectors)
        if config.freeze_embeddings:
            self.x_emb.weight.requires_grad = False

        # Encoder
        if config.q_cell == 'gru':
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
        self.q_mu = nn.Linear(q_d_last, config.d_z - self.y_dim)
        self.q_logvar = nn.Linear(q_d_last, config.d_z - self.y_dim)
        if self.y_dim > 0:
            self.q_lbl = nn.Linear(q_d_last, self.y_dim)
        else:
            self.q_lbl = None

        # Decoder
        if config.d_cell == 'gru':
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
            self.decoder
        ])

    def update_norm(self, min_v, max_v):
        self.min = P(torch.tensor(min_v, device=self.device), requires_grad=False)
        self.max = P(torch.tensor(max_v, device=self.device), requires_grad=False)

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

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

        return string

    def design_loss(self, z, y):
        dims = y.shape[1]
        designed_z = self.q_lbl(z)
        lbl_z = y
        nans = torch.isnan(y)
        lbl_z[nans] = designed_z[nans]
        return F.mse_loss(designed_z[~nans], y[~nans]) / dims, lbl_z
        # return F.mse_loss(designed_z, y) / dims, lbl_z

    def forward(self, x, y):
        """Do the VAE forward step

        :param x: list of tensors of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        # Encoder: x -> z, kl_loss
        z, kl_loss, mse_loss, _, _ = self.forward_encoder(x, y)

        # Decoder: x, z -> recon_loss
        recon_loss = self.forward_decoder(x, z)

        return kl_loss, recon_loss, mse_loss

    def forward_encoder(self, x, y=None):
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
        logvar = logvar.clamp(-1e-10, 10)
        z = mu + (logvar / 2).exp() * eps

        # Design: z -> labels, mse_loss
        if y is not None and not torch.all(torch.isnan(y)):
            mse_loss, lbl_z = self.design_loss(h, y)
            z = torch.cat((lbl_z, z), dim=1)
        else:
            mse_loss = torch.tensor([0.0], device=self.device)

        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()

        return z, kl_loss, mse_loss, mu, h

    def forward_decoder(self, x, z):
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        """

        lengths = [len(i_x) for i_x in x]

        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.pad)
        x_emb = self.x_emb(x)

        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True)

        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

        output, _ = self.decoder_rnn(x_input, h_0)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)

        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.pad
        )

        return recon_loss

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return torch.randn(n_batch, self.q_mu.out_features + self.y_dim,
                           device=self.x_emb.weight.device)

    def sample(self, n_batch, max_len=100, z=None, temp=1.0):
        """Generating n_batch samples in eval mode (`z` could be
        not on same device)

        :param n_batch: number of sentences to generate
        :param max_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param temp: temperature of softmax
        :return: list of tensors of strings, samples sequence x
        """
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(self.device)
            z_0 = z.unsqueeze(1)

            # Initial values
            h = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch,
                                                                    max_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len], device=self.device).repeat(
                n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.uint8,
                                   device=self.device).bool()

            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder_rnn(x_input, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)

                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.eos)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])

            return [self.tensor2string(i_x) for i_x in new_x]

    def design(self, properties, max_len=100, temp=1.0):
        """Generating samples based on properties in eval mode

        :param properties: nXm n samples that satisfied m properties
        :param max_len: max len of samples
        :param temp: temperature of softmax
        :return: list of tensors of strings, samples sequence x
        """
        with torch.no_grad():
            n_batch = properties.shape[0]
            z = self.sample_z_prior(n_batch)
            properties = properties - z[:, :properties.shape[1]]
            properties[torch.isnan(properties)] = 0.0
            z = torch.cat((properties, z), dim=1)

            return self.sample(n_batch, max_len=max_len, z=z, temp=temp)

    def recon(self, x_inputs, properties=None, max_len=100, temp=1.0,
              encode_times=10, decode_times=5):
        decode_results = []
        with torch.no_grad():
            for _ in range(encode_times):
                _, _, _, z_mean, h = self.forward_encoder(x_inputs)
                if self.q_lbl is not None:
                    lbl_z = self.q_lbl(h)
                    if properties is None:
                        properties = torch.zeros(*lbl_z.shape, device=self.device)
                    else:
                        properties -= lbl_z
                        properties[torch.isnan(properties)] = 0.0
                    z_mean = torch.cat((lbl_z + properties, z_mean), dim=1)
                if decode_times > 1:
                    z_mean = z_mean.repeat(decode_times, 1)
                # for _ in range(decode_times):
                decode_results.append(np.array(self.sample(z_mean.shape[0], max_len,
                                                           z=z_mean, temp=temp))
                                      .reshape((-1, decode_times), order='F'))
                del z_mean
                torch.cuda.empty_cache()
        return np.hstack(decode_results)

    def load_state_dict(self, state_dict, strict=True):
        """Resize mu and logvar"""

        if self.y_dim > 0:
            if  self.d_z - self.y_dim == state_dict['q_mu.bias'].shape[0]:
                self.load_norm = True
            else:
                self.load_norm = False
                lst = ['q_mu', 'q_logvar', 'encoder.1', 'encoder.2', 'vae.1.1', 'vae.1.2']
                for ele in lst:
                    state_dict[ele + '.bias'] = state_dict[ele + '.bias'][:self.d_z - self.y_dim]
                    state_dict[ele + '.weight'] = state_dict[ele + '.weight'][:self.d_z - self.y_dim, :]

                state_dict['min'] = self.min.data
                state_dict['max'] = self.max.data
                state_dict['q_lbl.bias'] = self.q_lbl.bias.data
                state_dict['q_lbl.weight'] = self.q_lbl.weight.data

        super(VAE, self).load_state_dict(state_dict, strict)
