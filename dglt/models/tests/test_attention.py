from unittest import TestCase


class TestAttention(TestCase):
    def test_reset_parameters(self):
        from dglt.models.layers import Attention
        import torch
        import numpy as np
        import numpy.testing as npt
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        atten = Attention(hidden=2,
                          in_feature=1,
                          out_feature=3)
        w1_test = np.array([[1.258218], [-0.2395837]])
        w2_test = np.array([[-1.3779874, 0.3595075],
                            [-0.68591213, -0.8845494],
                            [0.25509894,  0.5300144]])
        npt.assert_almost_equal(w1_test, atten.w1.detach().numpy())
        npt.assert_almost_equal(w2_test, atten.w2.detach().numpy())

    def test_forward(self):
        from dglt.models.layers import Attention
        import torch
        import numpy as np
        import numpy.testing as npt
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        atten = Attention(hidden=2,
                          in_feature=2,
                          out_feature=2)

        atten.w1 = torch.nn.Parameter(torch.FloatTensor([[0, 1], [1, 1]]))
        atten.w2 = torch.nn.Parameter(torch.FloatTensor([[1, 1], [0, 1]]))
        X = torch.FloatTensor([[1, 1], [0, 1]])
        out = np.array([[0.5504362, 1.],
                        [0.5504363, 1.]]
                      )
        # print(atten.forward(X))
        npt.assert_almost_equal(out, atten.forward(X)[0].detach().numpy())
