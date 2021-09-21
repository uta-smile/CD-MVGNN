from unittest import TestCase


class TestCompute_pnorm(TestCase):
    def test_compute_pnorm(self):
        import torch
        import numpy as np
        import dglt.models.nn_utils as dglt_nn_utils
        params = [np.array([[1.4, 2.8],
                            [-0.7, 9]], dtype=np.float32),
                  np.array([9.3, 6], dtype=np.float32)
                  ]
        linear_module = torch.nn.Linear(2, 2, bias=True)
        linear_module.weight = torch.nn.Parameter(torch.tensor(params[0]))
        linear_module.bias = torch.nn.Parameter(torch.tensor(params[1]))
        pnorm = dglt_nn_utils.compute_pnorm(linear_module)
        self.assertAlmostEqual(pnorm, np.sqrt(np.sum([np.linalg.norm(param)**2 for param in params])))



class TestCompute_gnorm(TestCase):
    def test_compute_gnorm(self):
        import torch
        import numpy as np
        import dglt.models.nn_utils as dglt_nn_utils
        w = torch.tensor([1, 2], dtype=torch.float32)
        b = torch.tensor(2, dtype=torch.float32)
        y = torch.tensor(6, dtype=torch.float32)
        x = torch.tensor([1, 2], dtype=torch.float32)
        grad_w = 2*(w.matmul(x)+b-y)*x
        grad_b = 2*(w.matmul(x)+b-y)

        model = torch.nn.Linear(2, 1, bias=True)
        model.weight = torch.nn.Parameter(torch.tensor(w))
        model.bias = torch.nn.Parameter(b)
        loss = torch.nn.MSELoss()(model(x), y)

        model.zero_grad()
        loss.backward()

        gnorm = dglt_nn_utils.compute_gnorm(model)
        self.assertAlmostEqual(gnorm, np.sqrt(np.sum([np.linalg.norm(g)**2 for g in [grad_w, grad_b]])))




class TestParam_count(TestCase):
    def test_param_count(self):
        self.fail()


class TestIndex_select_ND(TestCase):
    def test_index_select_ND(self):
        import torch
        import numpy as np
        import numpy.testing as npt
        import dglt.models.nn_utils as dglt_nn_utils
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        source = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
        index = torch.LongTensor([[0, 2], [1, 0]])
        res = dglt_nn_utils.index_select_ND(source, index)
        out = np.array([[[1., 2.], [5., 6.]],
                        [[3., 4.], [1., 2.]]])
        npt.assert_almost_equal(res.detach().numpy(), out)
        # print(out)


class TestConcat_input_features(TestCase):
    def test_concat_input_features(self):
        import dglt.models.nn_utils as dglt_nn_utils
        import torch
        import numpy as np
        import numpy.testing as npt
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        fea_batch = torch.FloatTensor([[1, 1], [0, 1]])
        mol_vecs = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
        res = dglt_nn_utils.concat_input_features(fea_batch, mol_vecs)
        out = np.array([[1, 2, 3, 1, 1], [4, 5, 6, 0, 1]])
        npt.assert_almost_equal(res.detach().numpy(), out)
        fea_batch = torch.FloatTensor([1, 1])
        mol_vecs = torch.FloatTensor([[1, 2, 3]])
        res = dglt_nn_utils.concat_input_features(fea_batch, mol_vecs)
        out = np.array([[1, 2, 3, 1, 1]])
        npt.assert_almost_equal(res.detach().numpy(), out)




class TestGet_activation_function(TestCase):
    def test_get_activation_function(self):
        self.fail()


class TestInitialize_weights(TestCase):
    def test_initialize_weights(self):
        self.fail()


class TestCompute_molecule_vectors(TestCase):
    def test_compute_molecule_vectors(self):
        self.fail()


class TestRalamb(TestCase):
    def test_get_lr(self):
        self.fail()

    def test_step(self):
        self.fail()


class TestLookahead(TestCase):
    def test_step(self):
        self.fail()


class TestRAdam(TestCase):
    def test_step(self):
        self.fail()


class TestNoamLR(TestCase):
    def test_get_lr(self):
        import torch
        import numpy as np
        import numpy.testing as npt
        from dglt.models.nn_utils import NoamLR
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        w1 = torch.nn.Parameter(torch.FloatTensor(1, 1))
        optimizer = torch.optim.Adam([w1])
        warmup_epochs = 2
        total_epochs = 10
        steps_per_epoch = 2
        init_lr = 1
        max_lr = 2
        final_lr = 1

        nlr = NoamLR(optimizer=optimizer,
                     warmup_epochs=warmup_epochs,
                     total_epochs=total_epochs,
                     steps_per_epoch=steps_per_epoch,
                     init_lr=init_lr,
                     max_lr=max_lr,
                     final_lr=final_lr)
        npt.assert_array_equal(np.array(nlr.get_lr()), np.array([1.0]))

    def test_step(self):
        import torch
        import numpy as np
        import numpy.testing as npt
        from dglt.models.nn_utils import NoamLR
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        w1 = torch.nn.Parameter(torch.FloatTensor(1, 1))
        w2 = torch.nn.Parameter(torch.FloatTensor(1, 1))

        optimizer = torch.optim.Adam([
            {'params': w1, 'lr': 1},
            {'params': w2, 'lr': 1}
        ])
        warmup_epochs = 2
        total_epochs = 10
        steps_per_epoch = 2
        init_lr = 1
        max_lr = 2
        final_lr = 1

        nlr = NoamLR(optimizer=optimizer,
                     warmup_epochs=warmup_epochs,
                     total_epochs=total_epochs,
                     steps_per_epoch=steps_per_epoch,
                     init_lr=init_lr,
                     max_lr=max_lr,
                     final_lr=final_lr)

        res = [[1.25],
               [1.5],
               [1.75],
               [2.0],
               [1.9152065613971474],
               [1.8340080864093427],
               [1.7562521603732997],
               [1.6817928305074294],
               [1.6104903319492547],
               [1.5422108254079414],
               [1.4768261459394998],
               [1.4142135623730956],
               [1.3542555469368933],
               [1.2968395546510103],
               [1.241857812073485],
               [1.189207115002722],
               [1.1387886347566925],
               [1.0905077326652586],
               [1.0442737824274146],
               [1.0000000000000009]]
        for i in range(total_epochs * steps_per_epoch):
            nlr.step()
            npt.assert_array_almost_equal(np.array(nlr.get_lr()), np.array(res[i] * 2))
