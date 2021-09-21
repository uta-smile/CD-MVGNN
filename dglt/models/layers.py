from argparse import Namespace
import math
import torch
from torch import nn as nn
from typing import Union

from dglt.models.nn_utils import index_select_ND, get_activation_function


class GaussianSampling(nn.Module):
    """sample elements based on hidden_layer"""

    def __init__(self,
                 input_size,
                 d_z,
                 bidirectional=False):
        """
        Initializes the GGaussianSampling

        :param input_size: the Input size
        :param d_z: The size of z
        :param num_layers: Number of layers
        :param bidirectional: if True bidirectional GRU is applied
        """
        super(GaussianSampling, self).__init__()
        d_last = input_size * (2 if bidirectional else 1)
        self.mu = nn.Linear(d_last, d_z)
        self.logvar = nn.Linear(d_last, d_z)

    def forward(self, h):
        mu = self.mu(h)
        logvar = self.logvar(h)

        eps = torch.randn_like(mu)
        logvar = logvar.clamp(-1e-10, 10)
        z = mu + (logvar / 2).exp() * eps

        return z, mu, logvar


class Attention(nn.Module):
    """
       Self Attention Layer
       Given $X\in \mathbb{R}^{n \times in_feature}$, the attention is calculated by: $a=Softmax(W_2tanh(W_1X))$, where
       $W_1 \in \mathbb{R}^{hidden \times in_feature}$, $W_2 \in \mathbb{R}^{out_feature \times hidden}$.
       The final output is: $out=aX$, which is unrelated with input $n$.
    """

    def __init__(self, *, hidden, in_feature, out_feature):
        """
        The init function.
        :param hidden: the hidden dimension, can be viewed as the number of experts.
        :param in_feature: the input feature dimension.
        :param out_feature: the output feature dimension.
        """
        super(Attention, self).__init__()
        self.w1 = torch.nn.Parameter(torch.FloatTensor(hidden, in_feature))
        self.w2 = torch.nn.Parameter(torch.FloatTensor(out_feature, hidden))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Use xavier_normal method to initialize parameters.
        """
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)

    def forward(self, X):
        """
        The forward function.
        :param X: The input feature map. $X \in \mathbb{R}^{n \times in_feature}$.
        :return: The final embeddings and attention matrix.
        """
        x = torch.tanh(torch.matmul(self.w1, X.transpose(1, 0)))
        x = torch.matmul(self.w2, x)
        attn = torch.nn.functional.softmax(x, dim=-1)
        x = torch.matmul(attn, X)
        return x, attn


class Readout(nn.Module):
    """The readout function. Convert node embeddings to graph embeddings."""

    def __init__(self,
                 rtype: str="none",
                 hidden_size: Union[int, float]=0,
                 attn_hidden: int=None,
                 attn_out: int=None,
                 args: Namespace=None,
                 ):
        """
        The readout function.
        :param rtype: readout type, can be "mean" and "self_attention".
        :param hidden_size: input hidden size
        :param attn_hidden: only valid if rtype == "self_attention". The attention hidden size.
        :param attn_out: only valid if rtype == "self_attention". The attention out size.
        :param args: legacy use.
        """
        super(Readout, self).__init__()
        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(int(hidden_size * 100)), requires_grad=False)
        self.rtype = "mean"

        # TODO: args should re-organized.
        if rtype == "self_attention":
            self.attn = Attention(hidden=attn_hidden,
                                  in_feature=int(hidden_size * 100),
                                  out_feature=attn_out)
            self.rtype = "self_attention"

    def forward(self, atom_hiddens, a_scope):
        # Readout
        mol_vecs = []
        self.attns = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                if self.rtype == "self_attention":
                    cur_hiddens, attn = self.attn(cur_hiddens)
                    cur_hiddens = cur_hiddens.flatten()
                    # Temporarily disable. Enable it if you want to save attentions.
                    # self.attns.append(attn.cpu().detach().numpy())
                else:
                    cur_hiddens = cur_hiddens.sum(dim=0) / a_size
                mol_vecs.append(cur_hiddens)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        return mol_vecs



class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 atom_messages: bool,
                 init_message_dim: int,
                 attached_fea_fdim: int,
                 hidden_size: int,
                 bias: bool,
                 depth: int,
                 dropout: float,
                 undirected: bool,
                 dense: bool,
                 aggregate_to_atom: bool,
                 attach_fea: bool,
                 activation: str,
                 input_layer='fc',
                 use_norm=False
                 ):
        """
        Initializes the MPNEncoder.
        :param args: legacy
        :param atom_messages: enables atom_messages or not
        :param init_message_dim:  the initial input message dimension
        :param attached_fea_fdim:  the attached feature dimension
        :param hidden_size: the output message dimension during message passing
        :param bias:
        :param depth: the message passing depth
        :param dropout:
        :param undirected: the message passing is undirected or not.
        :param dense: enables the dense connections.
        :param aggregate_to_atom: enable the output aggregation after message passing. If the bond message is enabled
        and aggregate to atom is disabled, the output of the encoder is num bonds x hidden.
        :param attach_fea: enables the feature attachment during message passing.
        :param use_norm: whether use layer norm or not.
        """
        super(MPNEncoder, self).__init__()
        self.init_message_dim = init_message_dim
        self.attached_fea_fdim = attached_fea_fdim
        self.input_layer = input_layer
        self.hidden_size = hidden_size
        self.bias = bias
        self.depth = depth
        self.dropout = dropout
        self.layers_per_message = 1
        self.undirected = undirected
        self.atom_messages = atom_messages
        self.dense = dense
        self.aggreate_to_atom = aggregate_to_atom
        self.attached_fea = attach_fea
        self.use_norm = use_norm

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(activation)

        # Input
        input_dim = self.init_message_dim
        if self.input_layer == 'fc':
            self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        else:
            self.gcn = GraphConvolution(input_dim, self.hidden_size, bias=self.bias)

        if self.attached_fea:
            w_h_input_size = self.hidden_size + self.attached_fea_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        if self.aggreate_to_atom:
            # TODO: dense support
            atom_fea_dim = self.init_message_dim if self.atom_messages else self.attached_fea_fdim
            self.W_o = nn.Linear(atom_fea_dim + self.hidden_size, self.hidden_size)

        if self.use_norm:
            self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)


    def forward(self,
                init_messages,
                init_attached_features,
                a2nei,
                a2attached,
                b2a=None,
                b2revb=None,
                adjs=None
                ) -> torch.FloatTensor:
        """

        :param init_messages:  initial massages, can be atom features or bond features.
        :param init_attached_features: initial attached_features.
        :param a2nei: the relation of item to its neighbors. For the atom message passing, a2nei = a2a. For bond
        messages a2nei = a2b
        :param a2attached: the relation of item to the attached features during message passing. For the atom message
        passing, a2attached = a2b. For the bond message passing a2attached = a2a
        :param b2a: remove the reversed bond in bond message passing
        :param b2revb: remove the revered atom in bond message passing
        :return: if aggreate_to_atom or self.atom_messages, return numatoms x hidden.
        Otherwise, return numbonds x hidden
        """

        # Input
        if self.input_layer == 'fc':
            input = self.W_i(init_messages)  # num_bonds x hidden_size # f_bond
        else:
            input = self.gcn(init_messages, adjs)

        attached_fea = init_attached_features # f_atom
        message = self.act_func(input)  # num_bonds x hidden_size
        msgs = [input]
        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                # two directions should be the same
                message = (message + message[b2revb]) / 2 # the same for atom and bond?

            nei_message = self.select_neighbor_and_aggregate(message, a2nei)
            a_message = nei_message
            if self.attached_fea:
                attached_nei_fea = self.select_neighbor_and_aggregate(attached_fea, a2attached)
                a_message = torch.cat((nei_message, attached_nei_fea), dim=1)

            if not self.atom_messages:
                rev_message = message[b2revb]
                if self.attached_fea:
                    atom_rev_message = attached_fea[b2a[b2revb]]
                    rev_message = torch.cat((rev_message, atom_rev_message), dim=1)
                # Except reverse bond its-self(w) ! \sum_{k\in N(u) \ w}
                message = a_message[b2a] - rev_message  # num_bonds x hidden
            else:
                message = a_message

            message = self.W_h(message)

            if self.dense:
                message = self.act_func(message)  # num_bonds x hidden_size
            else:
                message = self.act_func(input + message)

            message = self.dropout_layer(message)  # num_bonds x hidden


        output = message

        if self.aggreate_to_atom:
            # transfer bond/atom message to atom/bond message
            # bond message: atom to bond / atom_message: atom to atom
            a2x = a2nei
            atom_fea = init_messages if self.atom_messages else init_attached_features
            output = self.aggreate_to_atom_fea(message, a2x, atom_fea)

        return output  # num_atoms x hidden

    def aggreate_to_atom_fea(self, message, a2x, atom_fea):

        a_message = self.select_neighbor_and_aggregate(message, a2x)
        # do concat to atoms
        a_input = torch.cat([atom_fea, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)

        a_input = self.W_o(a_input)
        if self.use_norm:
            a_input = self.norm(a_input)
        atom_hiddens = self.act_func(a_input)  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        return atom_hiddens

    def select_neighbor_and_aggregate(self, feature, index):
        neighbor = index_select_ND(feature, index)
        return neighbor.sum(dim=1)


class GraphConvolution(nn.Module):
    """Simple GCN layer"""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Uniform weight and bias.
        :return:
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """

        :param input: input of model.
        :param adj: adjacency matrix.
        :return: output.
        """
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class MPNPlusEncoder(nn.Module):
    """A message passing neural network for encoding a molecule.
    Enables cross-dependent message passing between atom and bond.

    """

    def __init__(self,
                 atom_fea_dim: int,
                 bond_fea_fdim: int,
                 hidden_size: int,
                 bias: bool,
                 depth: int,
                 dropout: float,
                 undirected: bool,
                 dense: bool,
                 aggregate_to_atom: bool,
                 attach_fea: bool,
                 activation: str,
                 input_layer='fc'
                 ):
        """
        Initializes the encoder.
        :param args: legacy
        :param atom_fea_dim:  the atom feature dimension
        :param bond_fea_fdim:  the bond feature dimension
        :param hidden_size: the output message dimension during message passing
        :param bias:
        :param depth: the message passing depth
        :param dropout:
        :param undirected: the message passing is undirected or not.
        :param dense: enables the dense connections.
        :param aggregate_to_atom: enable the output aggregation after message passing. If the bond message is enabled
        and aggregate to atom is disabled, the output of the encoder is num bonds x hidden.
        :param attach_fea: enables the feature attachment during message passing.
        """
        super(MPNPlusEncoder, self).__init__()
        self.atom_fea_dim = atom_fea_dim
        self.bond_fea_fdim = bond_fea_fdim
        self.input_layer = input_layer
        self.hidden_size = hidden_size
        self.bias = bias
        self.depth = depth
        self.dropout = dropout
        self.layers_per_message = 1
        self.undirected = undirected
        self.dense = dense
        self.aggreate_to_atom = aggregate_to_atom
        self.attached_fea = attach_fea

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(activation)

        # Input
        # input_dim = self.atom_fea_dim
        if self.input_layer == 'fc':
            self.W_nin = nn.Linear(self.atom_fea_dim, self.hidden_size, bias=self.bias)
            self.W_ein = nn.Linear(self.bond_fea_fdim, self.hidden_size, bias=self.bias)
        else:
            self.gcn_nin = GraphConvolution(self.atom_fea_dim, self.hidden_size, bias=self.bias)
            self.gcn_ein = GraphConvolution(self.bond_fea_fdim, self.hidden_size, bias=self.bias)

        if self.attached_fea:
            w_node_input_size = self.hidden_size * 2 + self.bond_fea_fdim
            w_edge_input_size = self.hidden_size * 2 + self.atom_fea_dim
        else:
            w_node_input_size, w_edge_input_size = self.hidden_size * 2, self.hidden_size * 2

        # Shared weight matrix across depths (default)
        self.W_node = nn.Linear(w_node_input_size, self.hidden_size, bias=self.bias)
        self.W_edge = nn.Linear(w_edge_input_size, self.hidden_size, bias=self.bias)

        if self.aggreate_to_atom:
            # not share parameter between node and edge so far
            self.W_nout = nn.Linear(self.atom_fea_dim + self.hidden_size, self.hidden_size)
            self.W_eout = nn.Linear(self.atom_fea_dim + self.hidden_size, self.hidden_size)


    def forward(self,
                atom_features,
                bond_features,
                a2a,
                a2b,
                b2a=None,
                b2revb=None,
                adjs=None
                ) -> torch.FloatTensor:
        """

        :param atom_features:  initial massages, can be atom features or bond features.
        :param bond_features: initial attached_features.
        :param a2nei: the relation of item to its neighbors. For the atom message passing, a2nei = a2a. For bond
        messages a2nei = a2b
        :param a2attached: the relation of item to the attached features during message passing. For the atom message
        passing, a2attached = a2b. For the bond message passing a2attached = a2a
        :param b2a: remove the reversed bond in bond message passing
        :param b2revb: remove the revered atom in bond message passing
        :return: if aggreate_to_atom or self.atom_messages, return numatoms x hidden.
        Otherwise, return numbonds x hidden
        """

        # Input
        if self.input_layer == 'fc':
            atom_input = self.W_nin(atom_features)  # num_bonds x hidden_size # f_bond
            bond_input = self.W_ein(bond_features)
        else:
            atom_input = self.gcn_nin(atom_features, adjs)
            bond_input = self.gcn_ein(bond_features, adjs)

        # attached_fea = bond_features # f_atom
        atom_message = self.act_func(atom_input)  # num_bonds x hidden_size
        bond_message = self.act_func(bond_input)


        # Cross-dependent message passing
        for depth in range(self.depth - 1):
            atom_message = self.one_hop(message=atom_message,
                                        initial_message=atom_input,
                                        attached_message=bond_message,
                                        attached_features=bond_features,
                                        a2nei=a2a,
                                        a2attached=a2b,
                                        b2a=b2a,
                                        b2revb=b2revb,
                                        atom_messages=True)

            bond_message = self.one_hop(message=bond_message,
                                        initial_message=bond_input,
                                        attached_message=atom_message,
                                        attached_features=atom_features,
                                        a2nei=a2b,
                                        a2attached=a2a,
                                        b2a=b2a,
                                        b2revb=b2revb,
                                        atom_messages=False)

        if self.aggreate_to_atom:
            # transfer bond/atom message to atom/bond message
            # bond message: atom to bond / atom_message: atom to atom
            atom_message = self.aggreate_to_atom_fea(atom_message, a2a, atom_features, linear_module=self.W_nout)
            bond_message = self.aggreate_to_atom_fea(bond_message, a2b, atom_features, linear_module=self.W_eout)

        output = atom_message, bond_message

        return output  # 2 tuples, each one is: num_atoms x hidden

    def one_hop(self,
                message,
                initial_message,
                attached_message,
                attached_features,
                a2nei,
                a2attached,
                b2a,
                b2revb,
                atom_messages: bool = True):
        """

        :param message: message to be updated
        :param initial_message: initial message
        :param attached_message: message to be used for updating message.
        :param attached_features:
        :param a2nei:
        :param a2ttached:
        :param b2a:
        :param b2revb:
        :param atom_messages: whether it is atom message or not
        :return: hidden states after one hop
        """

        if self.undirected:
            # two directions should be the same
            message = (message + message[b2revb]) / 2  # the same for atom and bond?

        nei_message = self.select_neighbor_and_aggregate(message, a2nei)
        nei_attached_message = self.select_neighbor_and_aggregate(attached_message, a2attached)
        a_message = torch.cat((nei_message, nei_attached_message), dim=1)

        if self.attached_fea:
            attached_nei_fea = self.select_neighbor_and_aggregate(attached_features, a2attached)
            a_message = torch.cat((a_message, attached_nei_fea), dim=1)

        if not atom_messages:
            rev_message = torch.cat((message[b2revb], attached_message[b2a[b2revb]]), dim=1) #todo: need to confirm

            if self.attached_fea:
                atom_rev_message = attached_features[b2a[b2revb]]
                rev_message = torch.cat((rev_message, atom_rev_message), dim=1)
            # Except reverse bond its-self(w) ! \sum_{k\in N(u) \ w}
            message = a_message[b2a] - rev_message  # num_bonds x hidden
        else:
            message = a_message

        if atom_messages:
            message = self.W_node(message)
        else:
            message = self.W_edge(message)

        if self.dense:
            message = self.act_func(message)  # num_bonds x hidden_size
        else:
            message = self.act_func(initial_message + message)
        message = self.dropout_layer(message)  # num_bonds x hidden

        return message

    def aggreate_to_atom_fea(self, message, a2x, atom_fea, linear_module):

        a_message = self.select_neighbor_and_aggregate(message, a2x)
        # do concat to atoms
        a_input = torch.cat([atom_fea, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(linear_module(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        return atom_hiddens

    def select_neighbor_and_aggregate(self, feature, index):
        neighbor = index_select_ND(feature, index)
        return neighbor.sum(dim=1)
