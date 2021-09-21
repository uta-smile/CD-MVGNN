from argparse import ArgumentParser
from unittest import TestCase

import mock
from mock import patch, MagicMock, Mock
from dglt.models.modules import DualMPN
from dglt.data.featurization import BatchMolGraph, get_atom_fdim, get_bond_fdim
import dglt
import torch
from dglt.models.layers import MPNEncoder
from third_party.torchtest import torchtest as tt

class TestDualMPN(TestCase):

    def test_forward(self):
        import torch
        import numpy as np
        import numpy.testing as npt

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        with patch("dglt.models.modules.MPNEncoder") as mock_MPN, \
                patch("dglt.models.modules.Readout") as mock_Readout, \
                patch.object(dglt.models.modules, "get_atom_fdim") as mock_get_atom_fdim, \
                patch.object(dglt.models.modules, "get_atom_fdim") as mock_get_bond_fdim, \
                patch.object(dglt.models.modules, "concat_input_features") as mock_concat_input_feature:
            mock_MPN.forward.side_effect = lambda *args: args[0]
            # b = mock_Readout
            # b.side_effect = lambda *args: args[0]
            mock_concat_input_feature.side_effect = lambda *args: torch.cat([args[0], args[1]], dim=1)

            parser = ArgumentParser()
            args = parser.parse_args()
            args.features_only = False
            args.use_input_features = True
            args.no_attach_fea = True
            args.hidden_size = 1
            args.bias = True
            args.depth = 1
            args.dropout = 0
            args.undirected = False
            args.dense = False
            args.self_attention = False
            args.cuda = True
            args.attn_hidden = 1
            args.attn_out = 1

            f_atoms = torch.randn((20, 20), requires_grad=True)
            f_bonds = torch.randn((20, 20), requires_grad=True)
            features_bath = torch.randn((20, 20))

            batch = [(f_atoms,  # f_atoms
                      f_bonds,  # f_bonds
                      torch.zeros([20, 4], dtype=torch.long),  # a2b
                      torch.zeros(20, dtype=torch.long),  # b2a
                      torch.zeros(20, dtype=torch.long),  # b2revb
                      torch.LongTensor([[1, 14], [15, 5]]),
                      torch.LongTensor([[1, 14], [15, 5]]),  # a_scope, b_scope
                      torch.zeros([20, 4], dtype=torch.long),  # a2a
                      None),  # adjs_batch
                     features_bath  # features_batch
                     ]

            model = dglt.models.modules.DualMPN(args,
                                                atom_fdim=20,
                                                bond_fdim=20,
                                                graph_input=True,
                                                atom_emb_output=False)
            args.self_attention = True

            model = dglt.models.modules.DualMPN(args,
                                                atom_fdim=20,
                                                bond_fdim=20,
                                                graph_input=True,
                                                atom_emb_output=False)
            model.readout = Mock(side_effect=lambda *args: args[0])
            model.atom_encoder.forward = Mock(side_effect=lambda **kwargs: kwargs["init_messages"])
            model.edge_encoder.forward = Mock(side_effect=lambda **kwargs: kwargs["init_messages"])
            mol_edge_output, mol_atom_output = model.forward(batch=batch[0],
                                                             features_batch=batch[1])
            npt.assert_almost_equal(torch.cat([features_bath, f_bonds], dim=1).cpu().detach().numpy(),
                                    mol_edge_output.cpu().detach().numpy())
            npt.assert_almost_equal(torch.cat([features_bath, f_atoms], dim=1).cpu().detach().numpy(),
                                    mol_atom_output.cpu().detach().numpy())

            model.atom_emb_output = True
            atom_output, edge_output = model.forward(batch=batch[0], features_batch=batch[1])
            npt.assert_almost_equal(f_bonds.cpu().detach().numpy(),
                                    edge_output.cpu().detach().numpy())

            npt.assert_almost_equal(f_atoms.cpu().detach().numpy(),
                                    atom_output.cpu().detach().numpy())

            model.features_only = True
            features_output = model.forward(batch=batch[0], features_batch=batch[1])
            npt.assert_almost_equal(features_bath.cpu().detach().numpy(),
                                    features_output.cpu().detach().numpy())

            # res = model(args)
            # print(res)
