import torch
import torch.nn as nn
from torch.nn import functional as F
from ..modules import GRUEncoder, GRUDecoder
from ..layers import GaussianSampling


class VAE(nn.Module):
    """VAE model"""
    def __init__(self,
                 vocab,
                 encoder,
                 decoder,
                 sampling=None,
                 freeze_embeddings=False):
        """
        Initializes the VAE

        :param vocab: vocabulary for VAE
        :param encoder: the encoder module
        :param decoder: the decoder module
        :param sampling: the sampling layer. AutoEncoder if None
        :param freeze_embeddings: if True freeze the weights of embeddings
        """
        super(VAE, self).__init__()

        self.vocabulary = vocab

        # Special symbols
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(vocab, ss))

        # Word embeddings layer
        n_vocab, d_emb = len(vocab), vocab.vectors.size(1)
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.pad)
        self.x_emb.weight.data.copy_(vocab.vectors)
        if freeze_embeddings:
            self.x_emb.weight.requires_grad = False

        # model
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
        """Do the VAE forward step

        :param x: list of tensors of longs, input sentence x
        :return: tensor decoded
        """
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

        y = self.decoder(z, x_decoder, lengths)
        return y, mu, logvar

    def sample(self, n_batch, max_len=100, temp=1.0):
        new_x = self.decoder.sample(n_batch=n_batch, max_len=max_len, temp=temp, device=self.device)
        return new_x

    def get_loss_function(self):
        """define loss function"""
        def loss_func(input, target, mu=None, logvar=None, kl_weight=1.0):
            """The loss function

            :param input: the estimated input
            :param target: the target to be estimated
            :param mu: the mu of VAE
            :param logvar: the log var of VAE
            :param kl_weight: the kl hyper-parameter
            """
            target = nn.utils.rnn.pad_sequence(target, batch_first=True,
                                                      padding_value=self.pad)
            if mu is not None and logvar is not None:
                kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
            else:
                kl_loss = torch.tensor(0.0, device=self.device)
            recon_loss = F.cross_entropy(
                input[:, :-1].contiguous().view(-1, input.size(-1)),
                target[:, 1:].contiguous().view(-1),
                ignore_index=self.pad
            )

            return kl_weight * kl_loss + recon_loss, {'kl_loss':kl_loss,
                                                       'recon_loss': recon_loss}

        return loss_func

    @property
    def device(self):
        return next(self.parameters()).device


class moseVAE(VAE):
    def __init__(self, vocab, config):
        n_vocab, d_emb = len(vocab), vocab.vectors.size(1)
        encoder = GRUEncoder(d_emb, config.q_d_h,
                             num_layers=config.q_n_layers,
                             dropout=config.q_dropout,
                             bidirectional=config.q_bidir)
        decoder = GRUDecoder(d_emb, config.d_z, config.d_d_h, n_vocab,
                             num_layers=config.d_n_layers,
                             dropout=config.d_dropout)
        sample = GaussianSampling(config.q_d_h, config.d_z, bidirectional=config.q_bidir)
        super(moseVAE, self).__init__(vocab, encoder, decoder, sampling=sample)
        self.load_norm = False

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

        return string

    def string2tensor(self, string, device='model'):
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(
            ids, dtype=torch.long,
            device=self.device if device == 'model' else device
        )

        return tensor

    def update_norm(self, min_v, max_v):
        return None

    def forward(self, x, *args):
        output, mu, logvar = super(moseVAE, self).forward(x)

        loss_func = self.get_loss_function()
        _, loss_metric = loss_func(output, x, mu=mu, logvar=logvar)
        return loss_metric['kl_loss'], loss_metric['recon_loss'], torch.tensor([0.0], device=self.device)

    def sample(self, n_batch, max_len=100, temp=1.0):
        new_x = super(moseVAE, self).sample(n_batch, max_len=max_len, temp=temp)
        return self.tensor2string(new_x)
