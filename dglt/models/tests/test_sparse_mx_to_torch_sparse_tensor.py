from unittest import TestCase

class TestSparseMatrix(TestCase):
    def test_sparse_mx_to_torch_sparse_tensor(self):
        import numpy as np
        import numpy.testing as npt
        from dglt.data.transformer.utils import sparse_mx_to_torch_sparse_tensor

        adj = np.array([[0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 0]])
        sparse_tensor = sparse_mx_to_torch_sparse_tensor(adj)
        npt.assert_almost_equal(sparse_tensor.to_dense().detach().numpy(), adj)