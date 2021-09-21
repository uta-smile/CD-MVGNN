from unittest import TestCase
from mock import patch
from dglt.models.layers import GaussianSampling
import torch
import numpy as np
import numpy.testing as npt
from third_party.torchtest import torchtest as tt

class test_GaussianSampling(TestCase):

    def test_init_no_bidirectional(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        layer = GaussianSampling(2, 4, bidirectional=False)
        npt.assert_equal(np.array(layer.mu.weight.size()), np.array([4, 2]))
        npt.assert_equal(np.array(layer.mu.bias.size()), np.array([4]))
        npt.assert_equal(np.array(layer.logvar.weight.size()), np.array([4, 2]))
        npt.assert_equal(np.array(layer.logvar.bias.size()), np.array([4]))

    def test_init_bidirectional(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        layer = GaussianSampling(2, 5, bidirectional=True)
        npt.assert_equal(np.array(layer.mu.weight.size()), np.array([5, 4]))
        npt.assert_equal(np.array(layer.mu.bias.size()), np.array([5]))
        npt.assert_equal(np.array(layer.logvar.weight.size()), np.array([5, 4]))
        npt.assert_equal(np.array(layer.logvar.bias.size()), np.array([5]))

    def test_forward(self):
        from torch.nn import functional as F

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        layer = GaussianSampling(2, 5)

        batch = [(torch.randn(4, 2),), # input
                 (torch.randn((4, 5)),
                  torch.randn((4, 5)),
                  torch.randn((4, 5)))
                 ]

        z, mu, logvar = layer(*batch[0])
        npt.assert_almost_equal(
            np.array([[ 1.6892161,  0.3994248, -1.9570514, -0.7259943,  0.0348868],
                      [-1.1755203, -1.1872123, -0.8756769, -1.2386632, -0.4058160],
                      [ 0.6121122,  0.3426929, -1.0907680, -2.2619078, -1.5050428],
                      [-1.0973285,  2.3573747, -0.4234740, -0.0407090, -1.3532603]]),
            z.detach().numpy())
        npt.assert_almost_equal(
            np.array([[-0.0676502,  0.2220999, -0.3455490, -0.2465465, -0.1621414],
                      [ 0.3233159, -1.1165454, -0.5173218,  0.3228984, -0.0511727],
                      [-0.4689604,  0.1231683, -0.8434267, -0.8465199, -0.4263848],
                      [-0.3764187,  0.2104507, -0.6991090, -0.7074900, -0.3588707]]),
            mu.detach().numpy())
        npt.assert_almost_equal(
            np.array([[ 1.5377612e-01,  5.7804704e-02, -1.0000000e-10, -1.0000000e-10, -1.0000000e-10],
                      [ 4.8287681e-01,  1.3228369e-01, -1.0000000e-10, -1.0000000e-10, -1.0000000e-10],
                      [-1.0000000e-10,  1.0252492e+00, -1.0000000e-10, -1.0000000e-10, -1.0000000e-10],
                      [-1.0000000e-10,  7.5646925e-01, -1.0000000e-10, -1.0000000e-10, -1.0000000e-10]]),
            logvar.detach().numpy())
