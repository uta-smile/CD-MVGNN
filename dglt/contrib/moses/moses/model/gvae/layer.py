import torch
import torch.nn as nn

class RNN_Encoder(nn.Module):

    def __init__(self, vocab, config):
        super(RNN_Encoder, self).__init__()

        # get the vocabulary and configurations
        self.vocab = vocab
        self.config = config
        self.vocab_len = len(vocab) + 2
        self.ridx_bos = len(vocab) + 0
        self.ridx_pad = len(vocab) + 1

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

        # packe all modules into one
        self.encoder = nn.ModuleList([
            self.x_emb,
            self.encoder_rnn,
        ])

    def forward(self, x):
        x = [self.x_emb(v) for v in x]
        x = nn.utils.rnn.pack_sequence(x)
        __, h = self.encoder_rnn(x, None)
        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)

        return h

class VAE_Sample_Z(nn.Module):

    def __init__(self, vocab, config):
        super(VAE_Sample_Z, self).__init__()

        # get the vocabulary and configurations
        self.vocab = vocab
        self.config = config
        self.vocab_len = len(vocab) + 2 # BOS & EOS padding rules
        self.ridx_bos = len(vocab) + 0  # BOS-padding rule's index (exactly one)
        self.ridx_pad = len(vocab) + 1  # EOS-padding rule's index (zero, one, or more)

        # latent vector's mean & variance
        # q_d_last = self.config.q_d_h * (2 if self.config.q_bidir else 1)
        self.q_mu = nn.Linear(config.q_d_h, self.config.d_z)
        self.q_logvar = nn.Linear(config.q_d_h, self.config.d_z)

        # packe all modules into one
        self.sample_z = nn.ModuleList([
            self.q_mu,
            self.q_logvar
        ])

    def forward(self, h):
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)
        z = mu + (logvar / 2).exp() * torch.randn_like(mu)

        return z, mu, logvar

class RNN_Decoder(nn.Module):

    def __init__(self, vocab, config, x_emb=None):
        super(RNN_Decoder, self).__init__()

        # get the vocabulary and configurations
        self.vocab = vocab
        self.config = config
        self.vocab_len = len(vocab) + 2 # BOS & EOS padding rules
        self.ridx_bos = len(vocab) + 0  # BOS-padding rule's index (exactly one)
        self.ridx_pad = len(vocab) + 1  # EOS-padding rule's index (zero, one, or more)


        # x_emb
        if x_emb is None:
            self.x_emb = nn.Embedding(self.vocab_len, self.config.d_emb, self.ridx_pad)
            if self.config.freeze_embeddings:
                self.x_emb.weight.requires_grad = False

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
        self.decoder = nn.ModuleList([
            self.decoder_lat,
            self.decoder_rnn,
            self.decoder_fc
        ])

    def forward(self, x, z):
        lengths = [len(v) for v in x]
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.ridx_pad)
        x_emb = self.x_emb(x)
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths, batch_first=True)
        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
        output, __ = self.decoder_rnn(x_input, h_0)
        output, __ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)

        return y

