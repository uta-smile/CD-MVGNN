from unittest import TestCase
from third_party.torchtest import torchtest as tt
import itertools

class TestMPNEncoder(TestCase):
    def test_aggreate_to_atom_fea(self):
        from dglt.models.layers import MPNEncoder
        import torch
        import numpy as np
        import numpy.testing as npt

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        input_choices = [[False],
                         [3],
                         [3],
                         [3],
                         [False],
                         [1],
                         [0.1],
                         [False],
                         [False],
                         [True],
                         [False],
                         ['ReLU'],
                         ['fc']]
        mpns = [MPNEncoder(*i) for i in itertools.product(*input_choices)]
        for mpn in mpns:
            message = torch.FloatTensor([[1, 1, 1],
                                         [2, 2, 2],
                                         [3, 3, 3]])
            a2x = torch.tensor([[0],
                                [2],
                                [1]])
            atom_fea = torch.FloatTensor([[4, 4, 4],
                                          [5, 5, 5],
                                          [6, 6, 6]])
            output = mpn.aggreate_to_atom_fea(message, a2x, atom_fea)
            npt.assert_almost_equal(
                np.array([[0.0000, 1.111185, 0.0000], [0.0000, 0.0000, 0.0000], [0.0000, 1.0612156, 0.0000]], dtype=float),
                output.detach().numpy())

    def test_select_neighbor_and_aggregate(self):
        from dglt.models.layers import MPNEncoder
        import torch
        import numpy as np
        import numpy.testing as npt

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        input_choices = [[True, False],
                         [10],
                         [10],
                         [5],
                         [True, False],
                         [1],
                         [0.1],
                         [True, False],
                         [True, False],
                         [True, False],
                         [True, False],
                         ['ReLU'],
                         ['fc', 'gcn']]
        mpns = [MPNEncoder(*i) for i in itertools.product(*input_choices)]
        for mpn in mpns:
            message = torch.FloatTensor([[1, 1, 1],
                                         [2, 2, 2],
                                         [3, 3, 3]])
            index = torch.tensor([[0],
                                  [2],
                                  [1]])
            output = mpn.select_neighbor_and_aggregate(message, index)
            npt.assert_equal(np.array([[1, 1, 1], [3, 3, 3], [2, 2, 2]], dtype=float),
                             output.detach().numpy())

    def test_forward(self):
        from dglt.models.layers import MPNEncoder
        import torch
        from torch.nn import functional as F

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        input_choices = [[True, False],
                         [3],
                         [3],
                         [3],
                         [True, False],
                         [2],
                         [0.1],
                         [True, False],
                         [True, False],
                         [True, False],
                         [True, False],
                         ['ReLU'],
                         ['fc', 'gcn']]

        mpns = [MPNEncoder(*i) for i in itertools.product(*input_choices)]

        batch = [(torch.FloatTensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
                 torch.FloatTensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
                 torch.tensor([[0], [2], [1]]),
                 torch.tensor([[0], [2], [1]]),
                 torch.tensor([0, 1, 2]),
                 torch.tensor([0, 2, 1]),
                 torch.FloatTensor([[0, 0, 1], [1, 0, 0], [1, 0, 0]])),
                 torch.randn((3, 3))]

        for mpn in mpns:
            tt.test_suite(model=mpn,
                          loss_fn=F.mse_loss,
                          optim=torch.optim.Adam(mpn.parameters()),
                          batch=batch,
                          test_nan_vals=True,
                          device='cpu')
            tt.test_suite(model=mpn,
                          loss_fn=F.mse_loss,
                          optim=torch.optim.Adam(mpn.parameters()),
                          batch=batch,
                          test_inf_vals=True,
                          device='cpu')
