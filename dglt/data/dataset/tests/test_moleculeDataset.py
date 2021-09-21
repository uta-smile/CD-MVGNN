from unittest import TestCase
from mock import patch, MagicMock, Mock

from dglt.data.dataset.molecular import MoleculeDataset


class TestMoleculeDataset(TestCase):
    def test_compound_names(self):
        from dglt.data.dataset.molecular import MoleculeDatapoint
        from dglt.data.dataset.molecular import MoleculeDataset
        line = ['carbon_chain', 'CCCCC']
        data_point = MoleculeDatapoint(line, use_compound_names=True)
        dataset = MoleculeDataset([data_point])
        assert dataset.compound_names() == [line[0]]



    def test_smiles(self):
        from dglt.data.dataset.molecular import MoleculeDatapoint
        from dglt.data.dataset.molecular import MoleculeDataset
        line = ['CCCCC']
        data_point = MoleculeDatapoint(line)
        dataset = MoleculeDataset([data_point])
        assert dataset.smiles() == [line[0]]

    def test_features(self):
        from dglt.data.dataset.molecular import MoleculeDatapoint
        from dglt.data.dataset.molecular import MoleculeDataset
        import numpy as np
        line = ['CCCCC']
        data_point = MoleculeDatapoint(line)
        fea = np.array([1,2,3])
        data_point.set_features(fea)
        dataset = MoleculeDataset([data_point])
        assert dataset.features() == [fea]

    def test_adjs(self):
        from dglt.data.dataset.molecular import MoleculeDatapoint
        from dglt.data.dataset.molecular import MoleculeDataset
        import numpy.testing as npt
        import numpy as np
        import argparse
        lines = [['CC']]
        parser = argparse.ArgumentParser()
        parser.add_argument('--features_generator', type=str, default='')
        parser.add_argument('--input_layer', type=str, default='gcn')
        args = parser.parse_args()
        argss = [args, None]
        adjs = np.array([[0.5, 0.5],
                         [0.5, 0.5]])
        for i, args in enumerate(argss):
            mds = []
            for line in lines:
                mds.append(MoleculeDatapoint(line=line, args=args))
            dataset = MoleculeDataset(mds)
            if args is None:
                assert dataset.adjs() is None
            else:
                for a in dataset.adjs():
                    npt.assert_almost_equal(a, adjs)

    def test_targets(self):
        from dglt.data.dataset.molecular import MoleculeDatapoint
        from dglt.data.dataset.molecular import MoleculeDataset
        import numpy.testing as npt
        lines = [['CC'], ['CCC'], ['CCCC']]
        mds = []
        for line in lines:
            mds.append(MoleculeDatapoint(line))
        targets = [[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]]
        dataset = MoleculeDataset(mds)
        dataset.set_targets(targets)
        dataset_targets = dataset.targets()
        assert len(dataset_targets) == len(targets)
        for i in range(len(dataset_targets)):
            npt.assert_almost_equal(dataset_targets[i], targets[i])

    def test_num_tasks(self):
        self.fail()

    def test_features_size(self):
        self.fail()

    def test_shuffle(self):
        self.fail()

    def test_normalize_features(self):
        import torch
        from dglt.data.dataset.molecular import MoleculeDatapoint
        from dglt.data.dataset.molecular import MoleculeDataset
        import numpy as np
        import numpy.testing as npt
        from dglt.data.transformer.scaler import StandardScaler
        import argparse

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        smiless = [['FC(F)(F)C1COCO1'], ['N#CC=CCO'], ['CN1CC1CC(O)C#N']]

        #******************************************
        # ******test when features is None ********
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.input_layer = 'linear'
        args.features_generator = None
        args.num_bits = 4
        mols = []
        for smiles in smiless:
            mols.append(MoleculeDatapoint(line=smiles, args=args))
        dataset = MoleculeDataset(data=mols)
        self.assertEqual(dataset.normalize_features(scaler=None),  None)

        #********************************************************
        #******* test when the scalar argument is None *********
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.input_layer = 'linear'
        args.features_generator = ['morgan', 'morgan_count']
        args.num_bits = 4
        mols = []
        for smiles in smiless:
            mols.append(MoleculeDatapoint(line=smiles, args=args))
        dataset = MoleculeDataset(data=mols)
        dataset.normalize_features(scaler=None)
        npt.assert_almost_equal(np.vstack(dataset.features()), [[ 0., 0.,  0., 0., 1.29777137, 0., 1.41421356, -1.22474487],
                                                                [ 0., 0., 0., 0., -1.13554995, -1.22474487,-0.70710678, 0.],
                                                                [ 0.,0., 0., 0., -0.16222142, 1.22474487, -0.70710678,1.22474487]])
        self.assertIsInstance(dataset.scaler, StandardScaler)

        #*********************************************************************
        ##******* test normalize features with the scalar argument ***********
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.input_layer = 'linear'
        args.features_generator = ['morgan']
        args.num_bits = 9

        mols = []
        for smiles in smiless:
            mols.append(MoleculeDatapoint(line=smiles, args=args))
        dataset = MoleculeDataset(data=mols)

        scalar = StandardScaler(replace_nan_token=101)
        scalar.fit(np.vstack([d.features for d in dataset.data]))
        dataset.normalize_features(scaler=scalar)

        npt.assert_almost_equal(np.vstack(dataset.features()),
                                [[0.70710678, -1.41421356,  0., 0., 0.70710678, -0.70710678,  0., 0.70710678, 1.41421356],
                                 [0.70710678, 0.70710678, 0., 0., -1.41421356, -0.70710678, 0., -1.41421356, -0.70710678],
                                 [-1.41421356, 0.70710678, 0., 0., 0.70710678, 1.41421356, 0., 0.70710678, -0.70710678]])
        self.assertIsInstance(dataset.scaler, StandardScaler)



    def test_set_targets(self):
        from dglt.data.dataset.molecular import MoleculeDatapoint
        from dglt.data.dataset.molecular import MoleculeDataset
        import numpy.testing as npt
        import argparse
        lines = [['CC'], ['CCC'], ['CCCC']]
        parser = argparse.ArgumentParser()
        parser.add_argument('--features_generator', type=str, default='')
        parser.add_argument('--input_layer', type=str, default='linear')
        args = parser.parse_args()
        argss = [args, None]
        for args in argss:
            mds = []
            for line in lines:
                mds.append(MoleculeDatapoint(line=line, args=args))
            targets = [[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]
            dataset = MoleculeDataset(mds)
            dataset.set_targets(targets)
            assert len(dataset.data) == len(targets)
            for i in range(len(dataset.data)):
                npt.assert_almost_equal(dataset.data[i].targets, targets[i])

    def test_sort(self):
        import torch
        import numpy as np
        import numpy.testing as npt

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        with patch("dglt.data.dataset.molecular.MoleculeDatapoint") as mock_MD:
            mock_MD.args = None
            mock_MD.side_effect = lambda *args: int(args[0])
            ls = []
            for i in range(10):
                mmd = mock_MD(i)
                # mmd.data = i
                ls.append(mmd)

            from random import shuffle
            shuffle(ls)
            mds = MoleculeDataset(ls)
            mds.sort(lambda x: x)
            for d, i in enumerate(mds.data):
                self.assertEqual(i, d)




    def test_randomize_smiles(self):
        self.fail()

    def test_shuffle(self):
        import torch
        from dglt.data.dataset.molecular import MoleculeDatapoint
        from dglt.data.dataset.molecular import MoleculeDataset
        import argparse

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        smiless = [['FC(F)(F)C1COCO1'], ['N#CC=CCO'], ['CN1CC1CC(O)C#N']]

        parser = argparse.ArgumentParser()
        parser.add_argument('--features_generator', type=str, default='')
        parser.add_argument('--input_layer', type=str, default='linear')
        args = parser.parse_args()

        mols = []
        for smiles in smiless:
            mols.append(MoleculeDatapoint(line=smiles, args=args))
        dataset = MoleculeDataset(data=mols)

        dataset.shuffle(seed=3)
        self.assertEqual(len(dataset.data), len(smiless))
        self.assertEqual(dataset.data[0].smiles, smiless[1][0])
        self.assertEqual(dataset.data[1].smiles, smiless[2][0])
        self.assertEqual(dataset.data[2].smiles, smiless[0][0])

        dataset.shuffle(seed=None)
        self.assertEqual(dataset.data[0].smiles, smiless[1][0])
        self.assertEqual(dataset.data[1].smiles, smiless[0][0])
        self.assertEqual(dataset.data[2].smiles, smiless[2][0])

    def test__len__(self):
        import torch
        from dglt.data.dataset.molecular import MoleculeDatapoint
        from dglt.data.dataset.molecular import MoleculeDataset
        import argparse
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        # **** test when there are no data ****

        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.input_layer = 'linear'
        args.features_generator = ['morgan', 'morgan_count']
        args.num_bits = 3
        mols = []
        dataset = MoleculeDataset(data=mols)
        self.assertEqual(len(dataset), 0)

        # *** test when there are several data points
        smiless = [['FC(F)(F)C1COCO1'], ['N#CC=CCO'], ['CN1CC1CC(O)C#N']]
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.input_layer = 'linear'
        args.features_generator = ['morgan', 'morgan_count']
        args.num_bits = 3
        mols = []
        for smiles in smiless:
            mols.append(MoleculeDatapoint(line=smiles, args=args))
        dataset = MoleculeDataset(data=mols)
        self.assertEqual(len(dataset), 3)


