from unittest import TestCase

from dglt.data.dataset.molecular import MoleculeDatapoint


class TestMoleculeDatapoint(TestCase):
    def test_set_features(self):
        from dglt.data.dataset.molecular import MoleculeDatapoint
        import numpy as np
        import numpy.testing as npt
        line = ['CCCCC']
        data_point = MoleculeDatapoint(line)
        features = np.array([1, 2, 1])
        data_point.set_features(features)
        npt.assert_almost_equal(data_point.features,features)


    def test_num_tasks(self):
        import torch
        import numpy as np
        import numpy.testing as npt
        from argparse import ArgumentParser

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

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

        line = ["[Cl].CC(C)NCC(O)COc1cccc2ccccc12", 1, 2, 3]
        md = MoleculeDatapoint(line=line)
        self.assertEqual(md.num_tasks(), 3)

    def test_set_targets(self):
        from dglt.data.dataset.molecular import MoleculeDatapoint
        import numpy as np
        import numpy.testing as npt
        line = ['CCCCC']
        data_point = MoleculeDatapoint(line)
        target = list(np.array([1, 2, 1]))
        data_point.set_targets(target)
        npt.assert_almost_equal(data_point.targets, target)
