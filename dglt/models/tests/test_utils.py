from unittest import TestCase
import mock
from mock import patch, MagicMock, Mock
import dglt

from argparse import ArgumentParser
from dglt.models.utils import build_model

class TestBuild_finetune_model(TestCase):
    def test_build_finetune_model(self):
        self.fail()


class TestBuild_model(TestCase):
    def test_build_model(self):
        with patch("dglt.models.utils.GroverFinetuneTask") as mock_GroverFinetuneTask, \
                patch("dglt.models.utils.MPNN") as mock_MPNN, \
                patch("dglt.models.utils.MultiMPNN") as mock_MultiMPNN, \
                patch("dglt.models.utils.DualMPNN") as mock_DualMPNN:
            mock_MPNN.create_encoder.side_effect = lambda *args: 1
            mock_MPNN.create_ffn.side_effect = lambda *args: 1
            mock_MPNN.__str__.return_value = "MPNN"

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
            args.model_type = "dualmpnn"
            args.num_tasks = 2
            args.distinct_init = False
            args.dataset_type = None

            for type in ["mpnn", "dualmpnn", "multimpnn", "3dgcn", "grover"]:
                args.model_type = type
                model = build_model(args)
            args.model_type = ""
            ispass = False
            try:
                model = build_model(args)
            except NotImplementedError:
                ispass = True
            self.assertEqual(ispass, True)

    # self.fail()
