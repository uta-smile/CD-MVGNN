import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class SequentialEx(Module):
    "Like `nn.Sequential`, but with ModuleList semantics, and can access module input"

    def __init__(self, *layers): self.layers = nn.ModuleList(layers)

    def forward(self, x):
        res = x
        for l in self.layers:
            res.orig = x
            nres = l(res)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig = None
            res = nres
        return res

    def __getitem__(self, i): return self.layers[i]

    def append(self, l): return self.layers.append(l)

    def extend(self, l): return self.layers.extend(l)

    def insert(self, i, l): return self.layers.insert(i, l)


class MergeLayer(Module):
    "Merge a shortcut with the result of the module by adding them or concatenating thme if `dense=True`."

    def __init__(self, dense: bool = False):
        self.dense = dense

    def forward(self, x):
        return torch.cat([x, x.orig], dim=1) if self.dense else (x + x.orig)


def feed_forward(d_model: int, d_ff: int, ff_p: float = 0., act=nn.ReLU, double_drop: bool = True):
    layers = [nn.Linear(d_model, d_ff), act]
    if double_drop: layers.append(nn.Dropout(ff_p))
    return SequentialEx(*layers, nn.Linear(d_ff, d_model), nn.Dropout(ff_p), MergeLayer(), nn.LayerNorm(d_model))


class MultiHeadAttention(Module):
    "MutiHeadAttention."

    def __init__(self, n_heads: int, d_model: int, d_head: int = None, resid_p: float = 0., attn_p: float = 0.,
                 bias: bool = True,
                 scale: bool = True):
        d_head = d_head if d_head is not None else d_model // n_heads
        self.n_heads, self.d_head, self.scale = n_heads, d_head, scale
        self.attention = nn.Linear(d_model, 3 * n_heads * d_head, bias=bias)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=bias)
        self.drop_att, self.drop_res = nn.Dropout(attn_p), nn.Dropout(resid_p)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor = None, **kwargs):
        return self.ln(x + self.drop_res(self.out(self._apply_attention(x, mask=mask, **kwargs))))

    def _apply_attention(self, x: Tensor, mask: Tensor = None):
        bs, x_len = x.size(0), x.size(1)
        wq, wk, wv = torch.chunk(self.attention(x), 3, dim=-1)
        wq, wk, wv = map(lambda x: x.view(bs, x.size(1), self.n_heads, self.d_head), (wq, wk, wv))
        wq, wk, wv = wq.permute(0, 2, 1, 3), wk.permute(0, 2, 3, 1), wv.permute(0, 2, 1, 3)
        attn_score = torch.matmul(wq, wk)
        if self.scale: attn_score.div_(self.d_head ** 0.5)
        if mask is not None:
            minus_inf = -65504 if attn_score.dtype == torch.float16 else -1e9
            attn_score = attn_score.masked_fill(mask, minus_inf).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        attn_vec = torch.matmul(attn_prob, wv)
        return attn_vec.permute(0, 2, 1, 3).contiguous().contiguous().view(bs, x_len, -1)


def _line_shift(x: Tensor, mask: bool = False):
    "Shift the line i of `x` by p-i elements to the left, is `mask` puts 0s on the diagonal."
    bs, nh, n, p = x.size()
    x_pad = torch.cat([x.new_zeros(bs, nh, n, 1), x], dim=3)
    x_shift = x_pad.view(bs, nh, p + 1, n)[:, :, 1:].view_as(x)
    if mask: x_shift.mul_(torch.tril(x.new_ones(n, p), p - n)[None, None,])
    return x_shift


class DecoderLayer(Module):
    "Basic block of a Transformer model."

    # Can't use Sequential directly cause more than one input...
    def __init__(self, n_heads: int, d_model: int, d_head: int, d_inner: int, resid_p: float = 0., attn_p: float = 0.,
                 ff_p: float = 0.,
                 bias: bool = True, scale: bool = True, act=nn.ReLU, double_drop: bool = True,
                 attn_cls=MultiHeadAttention):
        self.mhra = attn_cls(n_heads, d_model, d_head, resid_p=resid_p, attn_p=attn_p, bias=bias, scale=scale)
        self.ff = feed_forward(d_model, d_inner, ff_p=ff_p, act=act, double_drop=double_drop)

    def forward(self, x: Tensor, mask: Tensor = None, **kwargs): return self.ff(self.mhra(x, mask=mask, **kwargs))


class Transformer(Module):
    "Transformer model: https://arxiv.org/abs/1706.03762."

    def __init__(self, n_layers: int, n_heads: int, d_model: int, d_head: int, d_inner: int,
                 resid_p: float = 0., attn_p: float = 0., ff_p: float = 0., embed_p: float = 0., bias: bool = True,
                 scale: bool = True,
                 act=nn.ReLU, double_drop: bool = True, attn_cls=MultiHeadAttention,
                 learned_pos_enc: bool = True, mask: bool = True, dense_out: bool = False, final_p: float = 0.):
        self.mask = mask
        self.drop_final = nn.Dropout(final_p)
        self.dense_out = dense_out
        self.layers = nn.ModuleList([DecoderLayer(n_heads, d_model, d_head, d_inner, resid_p=resid_p, attn_p=attn_p,
                                                  ff_p=ff_p, bias=bias, scale=scale, act=act, double_drop=double_drop,
                                                  attn_cls=attn_cls) for k in range(n_layers)])

    def reset(self):
        pass

    def forward(self, x, mask):
        inp = x
        if self.dense_out: out = x.new()
        for layer in self.layers:
            inp = layer(inp, mask=mask)
            inp_d = self.drop_final(inp)
            if self.dense_out: out = torch.cat([out, inp_d], dim=-1)
        return out if self.dense_out else inp_d


class AtomTransformer(Module):
    def __init__(self, n_layers, n_heads, d_model, embed_p: float = 0, final_p: float = 0, d_head=None,
                 deep_decoder=False,
                 dense_out=False, **kwargs):

        self.d_model = d_model
        d_head = d_head if d_head is not None else d_model // n_heads
        self.transformer = Transformer(n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_head=d_head,
                                       final_p=final_p, dense_out=dense_out, **kwargs)

        channels_out = d_model * n_layers if dense_out else d_model
        channels_out_scalar = channels_out + n_types + 1
        if deep_decoder:
            sl = [int(channels_out_scalar / (2 ** d)) for d in
                  range(int(math.ceil(np.log2(channels_out_scalar / 4) - 1)))]
            self.scalar = nn.Sequential(*(list(itertools.chain.from_iterable(
                [[nn.Conv1d(sl[i], sl[i + 1], 1), nn.ReLU(), nn.BatchNorm1d(sl[i + 1])] for i in range(len(sl) - 1)])) +
                                          [nn.Conv1d(sl[-1], 4, 1)]))
        else:
            self.scalar = nn.Conv1d(channels_out_scalar, 4, 1)

        self.magnetic = nn.Conv1d(channels_out, 9, 1)
        self.dipole = nn.Linear(channels_out, 3)
        self.potential = nn.Linear(channels_out, 1)

        self.pool = nn.AdaptiveAvgPool1d(1)

        n_atom_embedding = d_model // 2
        n_type_embedding = d_model - n_atom_embedding - 3  # - 1 - 1
        self.type_embedding = nn.Embedding(len(types) + 1, n_type_embedding)
        self.atom_embedding = nn.Embedding(len(atoms) + 1, n_atom_embedding)
        self.drop_type, self.drop_atom = nn.Dropout(embed_p), nn.Dropout(embed_p)

    def forward(self, xyz, type, ext, atom, mulliken, coulomb, mask_atoms, n_atoms):
        """
        :param xyz:  shape:(max_items, 3, max_atoms), num of molecular, 3, num of atoms
        :param type: shape: (max_items, 1, max_atoms)
        :param ext: max_items, 1, max_atoms
        :param atom: max_items, 1, max_atoms
        :param mulliken: max_items, 1, max_atoms
        :param coulomb:
        :param mask_atoms:
        :param n_atoms:
        :return:
        """
        bs, _, n_pts = xyz.shape
        t = self.drop_type(self.type_embedding((type + 1).squeeze(1)))
        a = self.drop_atom(self.atom_embedding((atom + 1).squeeze(1)))

        #        x = torch.cat([xyz, mulliken, ext, mask_atoms.type_as(xyz)], dim=1)
        # x = torch.cat([xyz, mask_atoms.type_as(xyz)], dim=1)
        x = xyz
        x = torch.cat([x.transpose(1, 2), t, a], dim=-1) * math.sqrt(self.d_model)  # B,N(29),d_model

        mask = (coulomb == 0).unsqueeze(1)
        x = self.transformer(x, mask).transpose(1, 2).contiguous()

        t_one_hot = torch.zeros(bs, n_types + 1, n_pts, device=type.device, dtype=x.dtype).scatter_(1, type + 1, 1.)

        scalar = self.scalar(torch.cat([x, t_one_hot], dim=1))
        magnetic = self.magnetic(x)
        px = self.pool(x).squeeze(-1)
        dipole = self.dipole(px)
        potential = self.potential(px)

        return type, ext, scalar, magnetic, dipole, potential

    def reset(self):
        pass
