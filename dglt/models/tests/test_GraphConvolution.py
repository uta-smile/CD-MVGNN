from unittest import TestCase
from mock import patch
from dglt.models.layers import GraphConvolution
import torch
import numpy as np
import numpy.testing as npt
from third_party.torchtest import torchtest as tt


class test_GraphConvolution(TestCase):

    @patch.object(torch, 'FloatTensor')
    @patch.object(GraphConvolution, 'reset_parameters')
    def test_init_with_bias(self, mock_reset_parameters, mock_FloatTensor):
        mock_FloatTensor.side_effect = lambda *args: torch.zeros(*args, dtype=torch.float32)
        mock_reset_parameters.return_value = None
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        gcn = GraphConvolution(in_features=1,
                               out_features=3)
        mock_reset_parameters.assert_called_once()
        npt.assert_almost_equal(np.zeros((1,3)), gcn.weight.detach().numpy())
        npt.assert_almost_equal(np.zeros((3,)), gcn.bias.detach().numpy())

    @patch.object(torch, 'FloatTensor')
    @patch.object(GraphConvolution, 'reset_parameters')
    def test_init_without_bias(self, mock_reset_parameters, mock_FloatTensor):
        mock_FloatTensor.side_effect = lambda *args: torch.zeros(*args, dtype=torch.float32)
        mock_reset_parameters.return_value = None
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        gcn = GraphConvolution(in_features=1,
                               out_features=3,
                               bias=False)
        mock_reset_parameters.assert_called_once()
        self.assertEqual(gcn.in_features, 1)
        self.assertEqual(gcn.out_features, 3)
        npt.assert_almost_equal(np.zeros((1, 3)), gcn.weight.detach().numpy())
        self.assertIsNone(gcn.bias)

    def test_reset_parameters_with_bias(self):
        from dglt.models.layers import GraphConvolution
        import torch
        import numpy as np
        import numpy.testing as npt

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        gcn = GraphConvolution(in_features=1,
                               out_features=3)
        self.assertEqual(gcn.in_features, 1)
        self.assertEqual(gcn.out_features, 3)
        npt.assert_almost_equal(np.array([[-0.0043225,  0.3097159, -0.4751853]], dtype=np.float32),
                                gcn.weight.detach().numpy())
        npt.assert_almost_equal(np.array([-0.4248946, -0.222369 ,  0.1548207], dtype=np.float32),
                                gcn.bias.detach().numpy())

    def test_reset_parameters_without_bias(self):
        from dglt.models.layers import GraphConvolution
        import torch
        import numpy as np
        import numpy.testing as npt

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        gcn = GraphConvolution(in_features=1,
                               out_features=3,
                               bias=False)
        self.assertEqual(gcn.in_features, 1)
        self.assertEqual(gcn.out_features, 3)
        npt.assert_almost_equal(np.array([[-0.0043225,  0.3097159, -0.4751853]], dtype=np.float32),
                                gcn.weight.detach().numpy())
        self.assertIsNone(gcn.bias)

    def test_extra_repr_with_bias(self):
        from dglt.models.layers import GraphConvolution
        import torch

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        gcn = GraphConvolution(in_features=1,
                               out_features=3)

        self.assertEqual(gcn.extra_repr(), 'in_features=1, out_features=3, bias=True')

    def test_extra_repr_without_bias(self):
        from dglt.models.layers import GraphConvolution
        import torch

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        gcn = GraphConvolution(in_features=1,
                               out_features=3,
                               bias=False)

        self.assertEqual(gcn.extra_repr(), 'in_features=1, out_features=3, bias=False')

    def test_forward_with_bias(self):
        from dglt.models.layers import GraphConvolution
        from torch.nn import functional as F

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        gcn = GraphConvolution(20, 120)

        batch = [(torch.randn(4, 20), # input
                 torch.sparse.FloatTensor(
                     torch.LongTensor([[0, 0, 1, 1, 2, 2, 3, 3],
                                       [0, 3, 1, 2, 1, 2, 0, 3]]),
                     torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1]),
                     torch.Size([4, 4]))),  # adj
                 torch.randn((4, 120))
                 ]

        tt.assert_vars_change(model=gcn,
                              loss_fn=F.mse_loss,
                              optim=torch.optim.Adam(gcn.parameters()),
                              batch=batch,
                              device='cpu'
                              )
        tt.test_suite(model=gcn,
                      loss_fn=F.mse_loss,
                      optim=torch.optim.Adam(gcn.parameters()),
                      batch=batch,
                      test_nan_vals=True,
                      device='cpu')
        tt.test_suite(model=gcn,
                      loss_fn=F.mse_loss,
                      optim=torch.optim.Adam(gcn.parameters()),
                      batch=batch,
                      test_inf_vals=True,
                      device='cpu')

        gcn = GraphConvolution(2, 5)

        batch = [(torch.randn(4, 2), # input
                  torch.sparse.FloatTensor(
                      torch.LongTensor([[0, 0, 1, 1, 2, 2, 3, 3],
                                        [0, 3, 1, 2, 1, 2, 0, 3]]),
                      torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1]),
                      torch.Size([4, 4]))),  # adj
                 torch.randn((4, 5))
                 ]

        output = gcn(*batch[0])
        npt.assert_almost_equal(
            np.array([[-0.43338  , -0.0026434,  0.2691842, -0.0560873, -0.6770933],
                      [ 0.4768988,  0.8264377,  0.0742766, -0.4757614,  0.5036826],
                      [ 0.4768988,  0.8264377,  0.0742766, -0.4757614,  0.5036826],
                      [-0.43338  , -0.0026434,  0.2691842, -0.0560873, -0.6770933]]),
            output.detach().numpy())

    def test_forward_without_bias(self):
        from dglt.models.layers import GraphConvolution
        from torch.nn import functional as F

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        gcn = GraphConvolution(20, 120, bias=False)

        batch = [(torch.randn(4, 20), # input
                  torch.sparse.FloatTensor(
                      torch.LongTensor([[0, 0, 1, 1, 2, 2, 3, 3],
                                        [0, 3, 1, 2, 1, 2, 0, 3]]),
                      torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1]),
                      torch.Size([4, 4]))),  # adj
                 torch.randn((4, 120))
                 ]

        tt.assert_vars_change(model=gcn,
                              loss_fn=F.mse_loss,
                              optim=torch.optim.Adam(gcn.parameters()),
                              batch=batch,
                              device='cpu'
                              )
        tt.test_suite(model=gcn,
                      loss_fn=F.mse_loss,
                      optim=torch.optim.Adam(gcn.parameters()),
                      batch=batch,
                      test_nan_vals=True,
                      device='cpu')
        tt.test_suite(model=gcn,
                      loss_fn=F.mse_loss,
                      optim=torch.optim.Adam(gcn.parameters()),
                      batch=batch,
                      test_inf_vals=True,
                      device='cpu')

        gcn = GraphConvolution(2, 5, bias=False)

        batch = [(torch.randn(4, 2), # input
                  torch.sparse.FloatTensor(
                      torch.LongTensor([[0, 0, 1, 1, 2, 2, 3, 3],
                                        [0, 3, 1, 2, 1, 2, 0, 3]]),
                      torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1]),
                      torch.Size([4, 4]))),  # adj
                 torch.randn((4, 5))
                 ]

        output = gcn(*batch[0])
        npt.assert_almost_equal(
            np.array([[-0.3510709, -0.0131681, -0.0762811,  0.1606063, -0.2225055],
                      [-1.3953472, -0.2705608, -0.2834834,  0.3674720, -0.8578523],
                      [-1.3953472, -0.2705608, -0.2834834,  0.3674720, -0.8578523],
                      [-0.3510709, -0.0131681, -0.0762811,  0.1606063, -0.2225055]]),
            output.detach().numpy())

