from unittest import TestCase
import itertools
from third_party.torchtest import torchtest as tt


class MPNTestArgs:
    def __init__(self, atom_messages, use_input_features, features_only,
                 hidden_size, bias, depth, dropout, undirected, dense,
                 self_attention, attn_hidden, attn_out, input_layer):
        self.atom_messages = atom_messages
        self.use_input_features = use_input_features
        self.features_only = features_only
        self.hidden_size = hidden_size
        self.bias = bias
        self.depth = depth
        self.dropout = dropout
        self.undirected = undirected
        self.dense = dense
        self.self_attention = self_attention
        self.attn_hidden = attn_hidden
        self.attn_out = attn_out
        self.cuda = False
        self.activation = 'ReLU'
        self.input_layer = input_layer



class TestMPN(TestCase):
    def test_forward(self):

        from dglt.models.modules import MPN
        import torch
        import numpy as np
        import torch.nn.functional as F

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        input_choices = [[True, False], [True], [False], [1],
                         [False], [2], [0.1], [True, False], [True, False], [True, False],
                         [1], [1], ['fc']]
        test_component = [MPN(MPNTestArgs(*i), 20, 20) for i in itertools.product(*input_choices)]

        batch = [((torch.randn((20, 20), requires_grad=True),  # f_atom
                   torch.randn((20, 20), requires_grad=True),  # f_bonds
                   torch.zeros([20, 4], dtype=torch.long),  # a2b
                   torch.zeros(20, dtype=torch.long),  # b2a
                   torch.zeros(20, dtype=torch.long),   # b2revb
                   [[1, 14], [15, 5]], [[1, 14], [15, 5]],  # a_scope, b_scope
                   torch.zeros([20, 4], dtype=torch.long),  # a2a
                   None), # adjs_batch
                   np.zeros((2, 20))  # adjs_batch
                  ),
                 torch.randn((2, 120))
                 ]

        for item in test_component:
            tt.assert_vars_change(model=item,
                                  loss_fn=F.mse_loss,
                                  optim=torch.optim.Adam(item.parameters()),
                                  batch=batch,
                                  device='cpu',
                                  params=[np for np in item.named_parameters() if 'cached_zero_vector' not in np[0]]
                                  )
            tt.test_suite(model=item,
                          loss_fn=F.mse_loss,
                          optim=torch.optim.Adam(item.parameters()),
                          batch=batch,
                          test_nan_vals=True,
                          device='cpu'
                          )
            tt.test_suite(model=item,
                          loss_fn=F.mse_loss,
                          optim=torch.optim.Adam(item.parameters()),
                          batch=batch,
                          test_inf_vals=True,
                          device='cpu'
                          )
