from unittest import TestCase
from third_party.torchtest import torchtest as tt

class TestReadout(TestCase):
    def test_forward(self):
        from dglt.models.layers import Readout
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        batch = [(torch.randn((20, 20),  requires_grad=True), [[1, 14], [15, 5]]),
                 torch.randn((2, 20), requires_grad=True)
                 ]

        test_component = [Readout(rtype='mean', hidden_size=2),
                          Readout(rtype='attn', hidden_size=2, attn_hidden=3, attn_out=2)]

        for item in test_component:
            if len(item._parameters) != 1:
                tt.assert_vars_change(model=item,
                                    loss_fn=F.mse_loss,
                                    optim=torch.optim.Adam(item.parameters()),
                                    batch=batch,
                                    device='cpu',
                                    params= [np for np in item.named_parameters() if np[0] is not 'cached_zero_vector']
                                    )

            tt.assert_vars_same(model=item,
                              loss_fn=F.mse_loss,
                              optim=torch.optim.Adam(item.parameters()),
                              batch=batch,
                              params=[('cached_zero_vector', item.cached_zero_vector)],
                              device='cpu'
                              )
            tt.test_suite(model=item,
                          loss_fn=F.mse_loss,
                          optim=torch.optim.Adam(item.parameters()),
                          batch=batch,
                          test_nan_vals=True,
                          device='cpu'
                          )
            tt.test_suite(model=item,
                          loss_fn=F.mse_loss,
                          optim=torch.optim.Adam(item.parameters()),
                          batch=batch,
                          test_inf_vals=True,
                          device='cpu'
                          )
