import torch
import torch.nn as nn
from torch.distributions.gumbel import Gumbel

from dglt.contrib.moses.moses.model.gvae.utils import Flatten

class CVAE(nn.Module):
    """Character Variational Autoencoder."""

    def __init__(self, vocab, config):
        """Constructor function.

        Args:
        * vocab: model's vocabulary
        * config: model's configuration
        """

        super().__init__()

        # get the vocabulary and configurations
        self.eos_symbol = '&'  # symbol '&' is not used in SMILES
        self.vocab = [self.eos_symbol] + vocab
        self.config = config
        self.vocab_len = len(self.vocab)

        # build encoder & decoder
        self.__build_encoder()
        self.__build_decoder()

        # pack all modules into one
        self.vae = nn.ModuleList([
            self.encoder,
            self.q_mu,
            self.q_logvar,
            self.mapper,
            self.decoder,
            self.decoder_fc
        ])

    def forward(self, x):
        """Perform the forward passing and compute losses.

        Args:
        * x: training samples (as torch.tensor)

        Returns:
        * kl_loss: KL-divergence loss
        * recon_loss: reconstruction loss
        """

        # send the data to model's device
        x = x.to(self.device)

        # encode into latent vectors
        x_trans = torch.transpose(x, 1, 2)
        x_encoded = self.encoder(x_trans)
        mu = self.q_mu(x_encoded)
        logvar = self.q_logvar(x_encoded)

        # decode from latent vectors
        z = mu + (logvar / 2).exp() * torch.randn_like(mu)
        z_mapped = self.mapper(z)
        z_tiled = z_mapped.unsqueeze(1).repeat(1, self.config.smiles_maxlen, 1)
        z_decoded, __ = self.decoder(z_tiled, None)
        y = self.decoder_fc(z_decoded)

        # compute KL-divergence & re-construction losses
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        recon_loss = -(x * y.log()).sum([1, 2]).mean()

        return kl_loss, recon_loss

    def sample(self, n_batch, max_len=100):
        """Sample SMILES strings from the prior distribution.

        Args:
        * n_batch: # of SMILES strings
        * max_len: maximal length of a SMILES string

        Returns:
        * string_list: list of SMILES strings
        """

        with torch.no_grad():
            # sample latent vectors from the prior distribution
            z = torch.randn(n_batch, self.config.d_z)
            z = z.to(self.device)

            # decode from latent vectors
            z_mapped = self.mapper(z)
            z_tiled = z_mapped.unsqueeze(1).repeat(1, max_len, 1)
            z_decoded, __ = self.decoder(z_tiled, None)
            y = self.decoder_fc(z_decoded)

            # convert tensors into SMILES strings
            m = Gumbel(torch.tensor([0.0]), torch.tensor([0.1]))
            noise = torch.squeeze(m.sample(y.size()))
            noise = noise.to(self.device)
            y_idxs = torch.argmax(y.log() + noise, dim=-1)
            tensor_list = torch.split(y_idxs, 1, dim=0)
            string_list = [self.tensor2string(tensor) for tensor in tensor_list]

        return string_list

    def string2tensor(self, string, device='model'):
        """Convert a SMILES string to torch.tensor.

        Args:
        * string: SMILES string
        * device: where to place the torch.tensor

        Returns:
        * tensor: torch.tensor consists of one-hot vectors
        """

        # obtain a list of non-zero entries' indices
        string += self.eos_symbol * (self.config.smiles_maxlen - len(string))
        ids = list(map(lambda x: self.vocab.index(x), string))

        # convert into a 2-D tensor consists of one-hot vectors
        tensor = torch.zeros(self.config.smiles_maxlen, self.vocab_len)
        tensor.scatter_(1, torch.tensor(ids).view([-1, 1]), 1)
        tensor.to(self.device if device == 'model' else device)

        return tensor

    def tensor2string(self, tensor):
        """Convert a torch.tensor to SMILES string.

        Args:
        * tensor: torch.tensor consists of non-zero entries' indices

        Returns:
        * string: SMILES string
        """

        # convert into a SMILES string with end-of-sequence characters removed
        ids = tensor.view(-1).tolist()
        string = ''.join([self.vocab[id] for id in ids])
        string = string.replace(self.eos_symbol, '')

        return string

    @property
    def device(self):
        """The model's device."""

        return next(self.parameters()).device

    def __build_encoder(self):
        """Build the encoder sub-network.

        NOTE: encoder's input must be of size <batch_size * vocab_len * smiles_maxlen>.
        """

        # configure parameters for convolutional and linear layers
        conv1_param = (self.vocab_len, 9, 9)  # in_channels / out_channels / kernel_size
        conv2_param = (9, 9, 9)
        conv3_param = (9, 10, 11)
        nb_idims_fc = conv3_param[1] * (self.config.smiles_maxlen -
            (conv1_param[2] - 1) - (conv2_param[2] - 1) - (conv3_param[2] - 1))
        nb_odims_fc = 435

        # encoder sub-network
        self.encoder = nn.Sequential(
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

    def __build_decoder(self):
        """Build the decoder sub-network."""

        # map the latent vector for decoding
        # <self.mapper>'s output should be repeated before feeding into <self.decoder>
        self.mapper = nn.Sequential(
            nn.Linear(self.config.d_z, self.config.d_z),
            nn.ReLU()
        )

        # decoder sub-network
        self.decoder = nn.GRU(
            self.config.d_z,
            self.config.d_d_h,
            num_layers=3,
            batch_first=True,
            dropout=self.config.d_dropout if self.config.d_n_layers > 1 else 0
        )

        # probabilistic outputs
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.config.d_d_h, self.vocab_len),
            nn.Softmax(dim=-1),
        )
