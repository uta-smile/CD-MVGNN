from argparse import Namespace
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from dglt.data.featurization import BatchMolGraph, get_atom_fdim, get_bond_fdim
from dglt.models.nn_utils import concat_input_features
from dglt.data.transformer.utils import sparse_mx_to_torch_sparse_tensor
from .layers import Readout, MPNEncoder, MPNPlusEncoder


class GRUEncoder(nn.Module):
    """An GRU based encoder"""

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0,
                 bidirectional=False):
        """
        Initializes the GRUCoder

        :param input_size: the Input size
        :param hidden_size: The Hidden size
        :param num_layers: Number of layers
        :param dropout: dropout rate
        :param bidirectional: if True bidirectional GRU is applied
        """
        super(GRUEncoder, self).__init__()
        self.encoder_rnn = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

    def forward(self, x):
        """Coder step, emulating x --> h

        :param x: tensors of longs, input sentence x
        :return: (n_batch, hidden_size * bidirectional) of floats, hidden layer
        :return: floats, forwarding all parameters
        """
        _, h = self.encoder_rnn(x, None)
        h = h[-(1+int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)
        return h


class GRUDecoder(nn.Module):
    """An GRU based encoder"""

    def __init__(self,
                 input_size,
                 d_z,
                 hidden_size,
                 output_size,
                 vocab,
                 num_layers=1,
                 dropout=0):
        """
        Initializes the GRUCoder

        :param input_size: the Input size
        :param d_z: the size of hidden z
        :param hidden_size: The Hidden size
        :param output_size: size of output
        :param num_layers: Number of layers
        :param dropout: dropout rate
        """
        super(GRUDecoder, self).__init__()
        # attributes
        self.d_z = d_z
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = num_layers

        # Special symbols
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(vocab, ss))

        self.decoder_rnn = nn.GRU(
            input_size + d_z,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0)
        self.decoder_lat = nn.Linear(d_z, hidden_size)
        self.decoder_fc = nn.Linear(hidden_size, output_size)

    def forward(self, z, x, lengths):
        """Coder step, emulating x --> h

        :param z: sampling z
        :param x: tensors of longs, input sentence x
        :param lengths: for training, length of each x
        :return: (n_batch, hidden_size * bidirectional) of floats, hidden layer
        """

        # reorder elements
        lengths = torch.tensor(lengths, device=x.device)
        is_order = torch.all(lengths[:-1] >= lengths[1:])
        if not is_order:
            order = torch.argsort(lengths, descending=True)
            reorder = torch.zeros_like(order)
            reorder.index_put_((order,), torch.arange(lengths.size(0), device=x.device))

            z = z[order]
            x = x[order]
            lengths = lengths[order]
        z_0 = z.unsqueeze(1).repeat(1, x.size(1), 1)
        x_input = torch.cat([x, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True)

        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.num_layers, 1, 1)

        output, _ = self.decoder_rnn(x_input, h_0)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        y = self.decoder_fc(output)
        if not is_order:
            return y[reorder]
        return y

    def sample(self, n_batch=1, z=None, max_len=100, temp=1.0, features=None, device=None,
               x_embedding=None):
        """Coder step, emulating x --> h

        :param n_batch: size of batch
        :param z: sampling z
        :param max_len: max length of sequence, Defaut 600
        :param temp: temperature of softmax
        :return: (n_batch, hidden_size * bidirectional) of floats, hidden layer
        """

        if device is None:
            device = self.decoder_lat.weight.device

        with torch.no_grad():
            if z is None:
                z = torch.randn(n_batch, self.d_z, device=device)
            else:
                n_batch = z.shape[0]
            z = z.to(device)
            z_0 = z.unsqueeze(1)

            h = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.bos, device=device).repeat(n_batch)
            x = torch.tensor([self.pad], device=device).repeat(n_batch, max_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len], device=device).repeat(n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.uint8, device=self.device).bool()

            # Generating cycle
            for i in range(1, max_len):
                x_emb = x_embedding(w).unsqueeze(1)
                if features is not None:
                    x_emb = torch.cat([x_emb, features.unsqueeze(1)], dim=-1)
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

        return new_x

    @property
    def device(self):
        return next(self.parameters()).device



class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False,
                 atom_emb_output=False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.use_input_features = args.use_input_features
        self.features_only = args.features_only
        self.atom_emb_output = atom_emb_output

        if args.atom_messages:
            init_message_dim = self.atom_fdim
            attached_fea_fdim = self.bond_fdim
        else:
            init_message_dim = self.bond_fdim
            attached_fea_fdim = self.atom_fdim

        self.encoder = MPNEncoder(args.atom_messages,
                                  init_message_dim,
                                  attached_fea_fdim,
                                  int(args.hidden_size * 100),
                                  args.bias,
                                  args.depth,
                                  args.dropout,
                                  args.undirected,
                                  args.dense,
                                  aggregate_to_atom=True,
                                  attach_fea=False,
                                  input_layer=args.input_layer,
                                  activation=args.activation
                                  )
        if args.self_attention:
            self.readout = Readout(rtype="self_attention", hidden_size=args.hidden_size, attn_hidden=args.attn_hidden,
                                   attn_out=args.attn_out, args=args)
        else:
            self.readout = Readout(rtype="mean", hidden_size=args.hidden_size, args=args)


    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None,
                ) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            if self.args.cuda:
                features_batch = features_batch.cuda()
            if self.features_only:
                return features_batch

        # mol_graph = mol2graph(smiles_batch, self.args)
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a, adjs_batch = batch
        # bond message: a2b b2a b2revb f_atoms f_bonds a_scope
        # atom message: a2a a2b b2revb f_atoms f_bonds a_scope
        # a2b atomidx -> list(bondidx) (n_atoms)
        # b2a bondidx -> atomidx (n_bonds)
        # a2a atomidx -> atomidx (n_atoms)
        # b2revb bondidx -> bondidx (n_bonds)
        #if self.atom_messages:

        if self.args.input_layer == 'gcn':
            adjs_batch = sparse_mx_to_torch_sparse_tensor(adjs_batch)

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

            #if self.atom_messages:
            a2a = a2a.cuda()
            if self.args.input_layer == 'gcn':
                adjs_batch = adjs_batch.cuda()

        if self.args.atom_messages:
            init_messages = f_atoms
            init_attached_features = f_bonds
            a2nei = a2a
            a2attached = a2b
        else:
            init_messages = f_bonds
            init_attached_features = f_atoms
            a2nei = a2b
            a2attached = a2a

        output = self.encoder.forward(init_messages, init_attached_features, a2nei, a2attached, b2a, b2revb, adjs_batch)

        # TODO: need more refactoring here.
        if not self.atom_emb_output:
            output = self.readout(output, a_scope)
            if self.args.use_input_features:
                output = concat_input_features(features_batch, output)
        return output


class DualMPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False,
                 atom_emb_output=False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        :param atom_emb_output: If true, output the atom embedding without readout.
        """
        super(DualMPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.use_input_features = args.use_input_features
        self.features_only = args.features_only
        self.atom_emb_output = atom_emb_output

        self.attach_fea = not args.no_attach_fea
        self.edge_encoder = MPNEncoder(atom_messages=False,
                                       init_message_dim=self.bond_fdim,
                                       attached_fea_fdim=self.atom_fdim,
                                       hidden_size=int(args.hidden_size * 100),
                                       bias=args.bias,
                                       depth=args.depth,
                                       dropout=args.dropout,
                                       undirected=args.undirected,
                                       dense=args.dense,
                                       aggregate_to_atom=True,
                                       attach_fea=self.attach_fea,
                                       activation=args.activation,
                                       use_norm=args.use_norm
                                       )
        self.atom_encoder = MPNEncoder(atom_messages=True,
                                       init_message_dim=self.atom_fdim,
                                       attached_fea_fdim=self.bond_fdim,
                                       hidden_size=int(args.hidden_size * 100),
                                       bias=args.bias,
                                       depth=args.depth,
                                       dropout=args.dropout,
                                       undirected=args.undirected,
                                       dense=args.dense,
                                       aggregate_to_atom=True,
                                       attach_fea=self.attach_fea,
                                       activation=args.activation,
                                       use_norm=args.use_norm
                                       )

        if args.self_attention:
            self.readout = Readout(rtype="self_attention", hidden_size=args.hidden_size, attn_hidden=args.attn_hidden,
                                   attn_out=args.attn_out, args=args)
        else:
            self.readout = Readout(rtype="mean", hidden_size=args.hidden_size, args=args)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None,
                adjs_batch: np.ndarray = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            if self.args.cuda:
                features_batch = features_batch.cuda()
            if self.features_only:
                return features_batch

        # mol_graph = mol2graph(smiles_batch, self.args)
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a, _ = batch
        a_scope = a_scope.data.cpu().numpy().tolist()
        b_scope = b_scope.data.cpu().numpy().tolist()
        # bond message: a2b b2a b2revb f_atoms f_bonds a_scope
        # atom message: a2a a2b b2revb f_atoms f_bonds a_scope
        # a2b atomidx -> list(bondidx) (n_atoms)
        # b2a bondidx -> atomidx (n_bonds)
        # a2a atomidx -> atomidx (n_atoms)
        # b2revb bondidx -> bondidx (n_bonds)
        # if self.atom_messages:
        # a2a = batch.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

            # if self.atom_messages:
            a2a = a2a.cuda()

        atom_output = self.atom_encoder.forward(init_messages=f_atoms,
                                                init_attached_features=f_bonds,
                                                a2nei=a2a,
                                                a2attached=a2b,
                                                b2a=b2a,
                                                b2revb=b2revb)

        edge_output = self.edge_encoder.forward(init_messages=f_bonds,
                                                init_attached_features=f_atoms,
                                                a2nei=a2b,
                                                a2attached=a2a,
                                                b2a=b2a,
                                                b2revb=b2revb)

        if self.atom_emb_output:
            return atom_output, edge_output

        # Share readout
        mol_edge_output = self.readout(edge_output, a_scope)
        mol_atom_output = self.readout(atom_output, a_scope)

        if self.args.use_input_features:
            mol_edge_output = concat_input_features(features_batch, mol_edge_output)
            mol_atom_output = concat_input_features(features_batch, mol_atom_output)
        return mol_edge_output, mol_atom_output



class DualMPNPlus(nn.Module):
    """
    A message passing neural network for encoding a molecule.
    Enabling the cross-dependent message passing between nodes and edges.
    """

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False,
                 atom_emb_output=False):
        """
        Initializes the model.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        :param atom_emb_output: If true, output the atom embedding without readout.
        """
        super(DualMPNPlus, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.use_input_features = args.use_input_features
        self.features_only = args.features_only
        self.atom_emb_output = atom_emb_output

        self.attach_fea = not args.no_attach_fea

        self.encoder = MPNPlusEncoder(atom_fea_dim=self.atom_fdim,
                                      bond_fea_fdim=self.bond_fdim,
                                      hidden_size=int(args.hidden_size * 100),
                                      bias=args.bias,
                                      depth=args.depth,
                                      dropout=args.dropout,
                                      undirected=args.undirected,
                                      dense=args.dense,
                                      aggregate_to_atom=True,
                                      attach_fea=self.attach_fea,
                                      activation=args.activation)

        if args.self_attention:
            self.readout = Readout(rtype="self_attention", hidden_size=args.hidden_size, attn_hidden=args.attn_hidden,
                                   attn_out=args.attn_out, args=args)
        else:
            self.readout = Readout(rtype="mean", hidden_size=args.hidden_size, args=args)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None,
                adjs_batch: np.ndarray = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            if self.args.cuda:
                features_batch = features_batch.cuda()
            if self.features_only:
                return features_batch

        # mol_graph = mol2graph(smiles_batch, self.args)
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a, _ = batch
        a_scope = a_scope.data.cpu().numpy().tolist()
        # b_scope = b_scope.data.cpu().numpy().tolist()
        # bond message: a2b b2a b2revb f_atoms f_bonds a_scope
        # atom message: a2a a2b b2revb f_atoms f_bonds a_scope
        # a2b atomidx -> list(bondidx) (n_atoms)
        # b2a bondidx -> atomidx (n_bonds)
        # a2a atomidx -> atomidx (n_atoms)
        # b2revb bondidx -> bondidx (n_bonds)
        # if self.atom_messages:
        # a2a = batch.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

            # if self.atom_messages:
            a2a = a2a.cuda()

        atom_output, edge_output = self.encoder(atom_features=f_atoms,
                                                bond_features=f_bonds,
                                                a2a=a2a,
                                                a2b=a2b,
                                                b2a=b2a,
                                                b2revb=b2revb)

        # atom_output = self.atom_encoder.forward(init_messages=f_atoms,
        #                                         init_attached_features=f_bonds,
        #                                         a2nei=a2a,
        #                                         a2attached=a2b,
        #                                         b2a=b2a,
        #                                         b2revb=b2revb)
        #
        # edge_output = self.edge_encoder.forward(init_messages=f_bonds,
        #                                         init_attached_features=f_atoms,
        #                                         a2nei=a2b,
        #                                         a2attached=a2a,
        #                                         b2a=b2a,
        #                                         b2revb=b2revb)

        if self.atom_emb_output:
            return atom_output, edge_output

        # Share readout
        mol_edge_output = self.readout(edge_output, a_scope)
        mol_atom_output = self.readout(atom_output, a_scope)

        if self.args.use_input_features:
            mol_edge_output = concat_input_features(features_batch, mol_edge_output)
            mol_atom_output = concat_input_features(features_batch, mol_atom_output)
        return mol_edge_output, mol_atom_output
