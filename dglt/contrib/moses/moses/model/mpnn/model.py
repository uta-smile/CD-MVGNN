from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mpn import MPN
from dglt.contrib.moses.moses.model.gvae.cvt_smiles_to_gidxs import Grammar
from dglt.contrib.moses.moses.model.gvae.layer import VAE_Sample_Z, RNN_Decoder

class MutiMPNN(nn.Module):
    def __init__(self, vocab, args):
        super(MutiMPNN, self).__init__()
        self.encoders = nn.ModuleList()
        if torch.device(args.device).type.startswith('cuda'):
            args.cuda = True

        self.config = args
        self.vocab = vocab
        self.vocab_len = len(vocab) + 2
        self.ridx_bos = len(vocab) + 0
        self.ridx_pad = len(vocab) + 1

        if args.nencoders == 1:
            encoder = MPN(args=args)
            self.encoders.append(encoder)
        else:
            input_encoder = MPN(args=args, atom_emb_output=True)
            self.encoders.append(input_encoder)
            for i in range(args.nencoders - 2):
                mid_encoder = MPN(args=args, bond_fdim=input_encoder.bond_fdim, atom_fdim=args.hidden_size*100, atom_emb_output=True)
                self.encoders.append(mid_encoder)

            #hard coding here
            out_encoders = MPN(args=args, bond_fdim=input_encoder.bond_fdim, atom_fdim=args.hidden_size*100, atom_emb_output=False)
            self.encoders.append(out_encoders)

        # compute output dim
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * 100
            if args.self_attention:
                first_linear_dim = first_linear_dim * args.attn_out

            if args.use_input_features:
                first_linear_dim += args.features_dim

        args.q_d_h = first_linear_dim

        # initialize grammar
        self.__init_grammar()

        # Append GVAE network
        self.sample_z = VAE_Sample_Z(vocab, args)
        self.decoder = RNN_Decoder(vocab, args)

        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def set_finetune_layers(self, args):
        for param in self.parameters():
            param.requires_grad = False

        self.ffn = self.create_ffn(args)
        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def forward(self, batch, features_batch, x):
        #TODO: Dense/Inception can apply here.
        for encoder in self.encoders:
            n_f_atom = encoder(batch, features_batch)
            batch.set_new_atom_feature(n_f_atom)
        mol_output = n_f_atom
        z, mu, logvar = self.sample_z(mol_output)
        y = self.decoder(x, z)
        y = y[:, :-1, :].contiguous()
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


# class RoyModel1(nn.Module):
#     def __init__(self, args):
#         super(RoyModel1, self).__init__()
#         dmpnn = MoleculeModel(classification=args.dataset_type=='classification')
#         self.encoder = dmpnn.create_encoder(args)
#         self.ffn = dmpnn.create_ffn(args)
#     def forward(self, e, x, addtional_feature):
#         x = torch.cat(e, x)
#         x = self.ffn(self.encoder(x, addtional_feature))
#         x = torch.cat(e, x)
#         x = self.ffn(self.encoder(x))
#         if self.classification and not self.training:
#             x = self.sigmoid(x)
#         return x
#
# class RoyModel2(nn.Module):
#     def __init__(self, args):
#         super(RoyModel2, self).__init__()
#         dmpnn = MoleculeModel(classification=args.dataset_type == 'classification')
#         self.encoder_dmpnn = dmpnn.create_encoder(args)
#         self.ffn_dmpnn = dmpnn.create_ffn(args)
#         args.atom_message = True
#         self.encoder_mpnn = dmpnn.create_encoder(args)
#         self.ffn_mpnn = dmpnn.create_encoder(args)
#
#     def forward(self, e, x, addtional_feature):
#         x1 = torch.cat(e, x)
#         x1 = self.ffn_dmpnn(self.encoder_dmpnn(x1, addtional_feature))
#
#         x2 = torch.cat(x, e)
#         x2 = self.ffn_mpnn(self.encoder_mpnn(x2, addtional_feature))
#
#         x1 = torch.cat(x1, x2)
#         x1 = self.ffn_dmpnn(self.encoder_dmpnn(x1))
#
#         x2 = torch.cat(x2, x1)
#         x2 = self.ffn_mpnn(self.encoder_mpnn(x2))
#
#         x1 = torch.nn.functional.max_pool1d(x1)
#         x2 = torch.nn.functional.max_pool1d(x2)
#         x = torch.cat(x1, x2)
#         return x
