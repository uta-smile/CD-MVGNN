from unittest import TestCase
from mock import patch
from dglt.models.modules import GRUDecoder
import torch
import numpy.testing as npt
import numpy as np

class Vocab():
    bos = 10
    eos = 11
    unk = 12
    pad = 13

class test_GRUDecoder(TestCase):

    def test_init_one_layer(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        vocab = Vocab()

        model = GRUDecoder(input_size=3,
                           d_z=2,
                           hidden_size=3,
                           output_size=4,
                           vocab=vocab,
                           num_layers=1,
                           dropout=0)

        npt.assert_equal(model.decoder_rnn.weight_ih_l0.size(), [9, 5])
        npt.assert_equal(model.decoder_rnn.weight_hh_l0.size(), [9, 3])
        npt.assert_equal(model.decoder_rnn.bias_ih_l0.size(), [9])
        npt.assert_equal(model.decoder_rnn.bias_hh_l0.size(), [9])
        npt.assert_equal(model.decoder_lat.weight.size(), [3, 2])
        npt.assert_equal(model.decoder_fc.weight.size(), [4, 3])


    def test_init_more_layers(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        vocab = Vocab()
        model = GRUDecoder(input_size=3,
                           d_z=2,
                           hidden_size=3,
                           output_size=4,
                           vocab=vocab,
                           num_layers=2,
                           dropout=0.3)

        npt.assert_equal(model.decoder_rnn.weight_ih_l0.size(), [9, 5])
        npt.assert_equal(model.decoder_rnn.weight_hh_l0.size(), [9, 3])
        npt.assert_equal(model.decoder_rnn.bias_ih_l0.size(), [9])
        npt.assert_equal(model.decoder_rnn.bias_hh_l0.size(), [9])
        npt.assert_equal(model.decoder_rnn.weight_ih_l1.size(), [9, 3])
        npt.assert_equal(model.decoder_rnn.weight_hh_l1.size(), [9, 3])
        npt.assert_equal(model.decoder_rnn.bias_ih_l1.size(), [9])
        npt.assert_equal(model.decoder_rnn.bias_hh_l1.size(), [9])
        npt.assert_equal(model.decoder_lat.weight.size(), [3, 2])
        npt.assert_equal(model.decoder_fc.weight.size(), [4, 3])

    def test_forward_one_layer(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        vocab = Vocab()
        model = GRUDecoder(input_size=3,
                           d_z=2,
                           hidden_size=3,
                           output_size=4,
                           vocab=vocab,
                           num_layers=1,
                           dropout=0)

        # seq_len = 1
        x = torch.tensor([[[1, 3, 9]]], dtype=torch.float32) # (1, 1,3)
        z = torch.tensor([[4, 7]], dtype=torch.float32) # (1,2)
        lengths = [1]
        y = model(z, x, lengths)
        npt.assert_almost_equal(y.detach().numpy(), np.array([[[0.489555, -0.3828769,  0.0424609,  0.9484077]]]))

    def test_forward_more_layers(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        vocab = Vocab()
        model = GRUDecoder(input_size=3,
                           d_z=2,
                           hidden_size=3,
                           output_size=4,
                           vocab=vocab,
                           num_layers=3,
                           dropout=0.5)

        # seq_len = 2
        x = torch.tensor([[[1, 3, 9], [2, 6, 8]]], dtype=torch.float32) #(1,2,3)
        z = torch.tensor([[4, 7]], dtype=torch.float32) #(1,2)
        lengths = [2]
        y = model(z, x, lengths)
        npt.assert_almost_equal(y.detach().numpy(), np.array([[[-0.6462525, -0.2237535,  0.7170722, -0.3933115],
                                                             [-0.5742503, -0.2249022,  0.5865318, -0.2882331]]]))
    def test_forward_out_of_order(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        vocab = Vocab()
        model = GRUDecoder(input_size=3,
                           d_z=2,
                           hidden_size=3,
                           output_size=4,
                           vocab=vocab,
                           num_layers=1,
                           dropout=0)

        # seq_len = 1
        x = torch.tensor([[[4, 5, 6], [13, 13, 13]], [[1, 3, 9], [3, 4, 5]]], dtype=torch.float32) # (1, 1,3)
        z = torch.tensor([[4, 7], [5, 6]], dtype=torch.float32) # (1,2)
        lengths = [1, 2]
        y = model(z, x, lengths)
        npt.assert_almost_equal(y.detach().numpy(), np.array([[[-0.02034739,  1.379099  ,  1.0464997 ,  1.8012743 ],
                                                               [ 0.3425867 , -0.3513402 ,  0.52387035,  0.39565808]],
                                                              [[ 0.92926306, -0.38100234, -0.20549005,  1.0010636 ],
                                                               [ 1.5928643 , -0.16080213, -0.29571295,  0.9129977 ]]]))

    def test_forward_inorder(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        vocab = Vocab()
        model = GRUDecoder(input_size=3,
                           d_z=2,
                           hidden_size=3,
                           output_size=4,
                           vocab=vocab,
                           num_layers=1,
                           dropout=0)

        # seq_len = 1
        x = torch.tensor([[[1, 3, 9], [3, 4, 5]], [[4, 5, 6], [13, 13, 13]]], dtype=torch.float32) # (1, 1,3)
        z = torch.tensor([[5, 6], [4, 7]], dtype=torch.float32) # (1,2)
        lengths = [2, 1]
        y = model(z, x, lengths)
        npt.assert_almost_equal(y.detach().numpy(), np.array([[[ 0.92926306, -0.38100234, -0.20549005,  1.0010636 ],
                                                               [ 1.5928643 , -0.16080213, -0.29571295,  0.9129977 ]],
                                                              [[-0.02034739,  1.379099  ,  1.0464997 ,  1.8012743 ],
                                                               [ 0.3425867 , -0.3513402 ,  0.52387035,  0.39565808]]]))

    @patch.object(torch, 'multinomial')
    def test_sample_batch_nofeature(self, mock_multinomial):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        mock_multinomial.side_effect = [torch.tensor([[12]]),
                                        torch.tensor([[12]]),
                                        torch.tensor([[11]])]
        vocab = Vocab()
        model = GRUDecoder(input_size=4,
                           d_z=2,
                           hidden_size=3,
                           output_size=4,
                           vocab=vocab,
                           num_layers=1,
                           dropout=0)

        x_emb = lambda x: torch.tensor([[0.5, 0.4, 0.6, 1]])
        x = model.sample(n_batch=1, max_len=3, x_embedding=x_emb)
        npt.assert_equal(x[0].detach().numpy(), np.array([10, 12, 12]))

    @patch.object(torch, 'multinomial')
    def test_sample_batch_feature(self, mock_multinomial):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        mock_multinomial.side_effect = [torch.tensor([[12]]),
                                        torch.tensor([[12]]),
                                        torch.tensor([[11]])]
        vocab = Vocab()
        model = GRUDecoder(input_size=4,
                           d_z=2,
                           hidden_size=3,
                           output_size=4,
                           vocab=vocab,
                           num_layers=1,
                           dropout=0)

        x_emb = lambda x: torch.tensor([[0.5, 0.4]])
        feature = torch.tensor([[0.23, 0.45]])
        x = model.sample(n_batch=1, features=feature, max_len=3, x_embedding=x_emb)
        npt.assert_equal(x[0].detach().numpy(), np.array([10, 12, 12]))

    @patch.object(torch, 'multinomial')
    def test_sample_batch_toeos(self, mock_multinomial):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        mock_multinomial.side_effect = [torch.tensor([[12]]),
                                        torch.tensor([[12]]),
                                        torch.tensor([[11]]),
                                        torch.tensor([[12]])]
        vocab = Vocab()
        model = GRUDecoder(input_size=4,
                           d_z=2,
                           hidden_size=3,
                           output_size=4,
                           vocab=vocab,
                           num_layers=1,
                           dropout=0)

        x_emb = lambda x: torch.tensor([[0.5, 0.4]])
        feature = torch.tensor([[0.23, 0.45]])
        x = model.sample(n_batch=1, features=feature, max_len=5,  x_embedding=x_emb)
        npt.assert_equal(x[0].detach().numpy(), np.array([10, 12, 12, 11]))

    @patch.object(torch, 'multinomial')
    def test_sample_batch_toeos(self, mock_multinomial):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        mock_multinomial.side_effect = [torch.tensor([[12]]),
                                        torch.tensor([[12]]),
                                        torch.tensor([[11]]),
                                        torch.tensor([[12]])]
        vocab = Vocab()
        model = GRUDecoder(input_size=4,
                           d_z=2,
                           hidden_size=3,
                           output_size=4,
                           vocab=vocab,
                           num_layers=1,
                           dropout=0)

        x_emb = lambda x: torch.tensor([[0.5, 0.4]])
        feature = torch.tensor([[0.23, 0.45]])
        z = torch.tensor([[1.0, 4.5]])
        x = model.sample(z=z, features=feature, max_len=5,  x_embedding=x_emb)
        npt.assert_equal(x[0].detach().numpy(), np.array([10, 12, 12, 11]))