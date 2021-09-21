import numpy as np
import torch
import torch.nn as nn

from dglt.data.featurization.mol2graph import get_atom_fdim
from dglt.models.layers import Readout
from dglt.models.nn_utils import index_select_ND


class TDGCN(nn.Module):
    def __init__(self, args):
        super(TDGCN, self).__init__()
        init_atom_dim = get_atom_fdim(args)
        hidden_dim = 1100
        ffn_hidden_dim = 700
        self.args = args
        self.coord = args.coord
        self.dropout_layer = nn.Dropout(p=0.15)
        self.acts = nn.ReLU()

        self.W_is = nn.Linear(init_atom_dim, hidden_dim, bias=False)
        self.W_iv = nn.Linear(init_atom_dim * 2, hidden_dim, bias=False)

        self.W_hs = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_hv = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_os = nn.Linear(init_atom_dim + hidden_dim, ffn_hidden_dim)
        self.W_ov = nn.Linear(hidden_dim * 2, ffn_hidden_dim)

        self.readout = Readout()

        if self.args.use_input_features:
            ffn_input_dim = ffn_hidden_dim + 200
        else:
            ffn_input_dim = ffn_hidden_dim * 2

        self.ffn = nn.Sequential(self.dropout_layer, nn.Linear(ffn_input_dim, ffn_hidden_dim), self.acts,
                                 self.dropout_layer, nn.Linear(ffn_hidden_dim, 1))
        

    def forward(self, batch, features_batch):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
        if self.args.use_input_features:
            features_batch = torch.from_numpy(np.array(features_batch)).float().cuda()
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = \
            f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(),\
            b2revb.cuda(), a_scope.cuda(), b_scope.cuda(), a2a.cuda()
        si = self.W_is(f_atoms)
        s = self.acts(si)

        s_nei_atom = index_select_ND(f_atoms, a2a)
        self_loop = f_atoms.unsqueeze(1) * (a2a > 0).float().unsqueeze(-1)
        s_nei_atom = torch.cat([s_nei_atom, self_loop], -1)
        atom_nei_bond = index_select_ND(f_bonds, a2b)
        atom_nei_bond = atom_nei_bond.unsqueeze(-1)
        vi = (s_nei_atom.unsqueeze(-2) * atom_nei_bond).sum(1)
        vi = self.W_iv(vi)
        v = self.acts(vi)
        norm_factor = (a2a > 0).sum(1)
        norm_factor[norm_factor == 0] = 1

        for i in range(6):
            # s to v
            s = index_select_ND(s, a2a).sum(1)
            v = index_select_ND(v, a2a).sum(1) / norm_factor.view(norm_factor.shape[0], 1, 1).float()

            s = self.dropout_layer(self.acts(si + self.W_hs(s)))
            v = self.dropout_layer(self.acts(vi + self.W_hv(v)))

        os = index_select_ND(s, a2a).sum(1)
        os = torch.cat([f_atoms, os], dim=-1)
        os = self.dropout_layer(self.acts(self.W_os(os)))

        ov = index_select_ND(v, a2a).sum(1)
        ov = torch.cat([vi, ov], dim=-1)
        ov = self.dropout_layer(self.acts(self.W_ov(ov)))

        # v to s
        # v_nei_atom = index_select_ND(v, a2a)
        # v_nei_atom = v_nei_atom.view(-1, v_nei_atom.shape[-2], v_nei_atom.shape[-1])
        # v_nei_atom = v_nei_atom.transpose(1, 2)
        # atom_nei_bond = atom_nei_bond.view(-1, atom_nei_bond.shape[-2], atom_nei_bond.shape[-1])
        # v2s = torch.bmm(v_nei_atom, atom_nei_bond)
        # v2s = v2s.squeeze(-1).view(s.shape[0], -1, s.shape[1]).sum(1)

        # ov = torch.cat([f_atoms, v2s], dim=-1)
        # ov = self.dropout_layer(self.acts(self.W_ov(ov)))

        o = torch.cat([os, ov.sum(1)], dim=1)
        o = self.readout(o, a_scope)

        if self.args.use_input_features:
            o = torch.cat([o, features_batch], dim=-1)
        o = self.ffn(o)
        return o





