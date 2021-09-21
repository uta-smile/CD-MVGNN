from unittest import TestCase
from dglt.models.modules import GRUEncoder
import torch
import numpy.testing as npt
import numpy as np

class test_GRUEncoder(TestCase):

    def test_init_one_layer(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = GRUEncoder(input_size=2,
                           hidden_size=3,
                           num_layers=1,
                           dropout=0.3)

        npt.assert_equal(model.encoder_rnn.weight_ih_l0.size(), [9, 2])
        npt.assert_equal(model.encoder_rnn.weight_hh_l0.size(), [9, 3])
        npt.assert_equal(model.encoder_rnn.bias_ih_l0.size(), [9])
        npt.assert_equal(model.encoder_rnn.bias_hh_l0.size(), [9])
        npt.assert_equal(model.encoder_rnn.dropout, 0.0)

    def test_init_more_layers(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = GRUEncoder(input_size=2,
                           hidden_size=3,
                           num_layers=2,
                           dropout=0.3)

        npt.assert_equal(model.encoder_rnn.weight_ih_l0.size(), [9, 2])
        npt.assert_equal(model.encoder_rnn.weight_hh_l0.size(), [9, 3])
        npt.assert_equal(model.encoder_rnn.bias_ih_l0.size(), [9])
        npt.assert_equal(model.encoder_rnn.bias_hh_l0.size(), [9])
        npt.assert_equal(model.encoder_rnn.weight_ih_l1.size(), [9, 3])
        npt.assert_equal(model.encoder_rnn.weight_hh_l1.size(), [9, 3])
        npt.assert_equal(model.encoder_rnn.bias_ih_l1.size(), [9])
        npt.assert_equal(model.encoder_rnn.bias_hh_l1.size(), [9])
        npt.assert_equal(model.encoder_rnn.dropout, 0.3)

    def test_forward_one_layer(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        x = torch.tensor([[[1, 3]]], dtype=torch.float32)  # shape: (1, 1, 2)

        model_bidirectional_false = GRUEncoder(input_size=2,
                                               hidden_size=3,
                                               num_layers=1,
                                               dropout=0.3,
                                               bidirectional=False)
        y = model_bidirectional_false(x)
        npt.assert_almost_equal(y.detach().numpy(), np.array([[-0.2497127,  0.0453912,  0.4915223]]))

        model_bidirectional_true = GRUEncoder(input_size=2,
                                              hidden_size=2,
                                              num_layers=1,
                                              dropout=0.7,
                                              bidirectional=True)
        y = model_bidirectional_true(x)
        npt.assert_almost_equal(y.detach().numpy(), np.array([[ 0.2764624,  0.5799975, -0.7154059, -0.647683 ]]))

    def test_forward_more_layers(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        x = torch.tensor([[[1, 3], [2, 6]]], dtype=torch.float32)  # shape: (1, 2, 2)

        model_bidirectional_false = GRUEncoder(input_size=2,
                                               hidden_size=3,
                                               num_layers=2,
                                               dropout=0.3,
                                               bidirectional=False)
        y = model_bidirectional_false(x)
        npt.assert_almost_equal(y.detach().numpy(), np.array([[-0.4384642, -0.5577565, -0.3396689]]))

        model_bidirectional_true = GRUEncoder(input_size=2,
                                              hidden_size=3,
                                              num_layers=6,
                                              dropout=0.9,
                                              bidirectional=True)
        y = model_bidirectional_true(x)
        npt.assert_almost_equal(y.detach().numpy(), np.array([[0.4214117, -0.4436734, -0.7983335,
                                                                0.0196403,  0.0363605, 0.2752606]]))