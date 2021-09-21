import torch
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam
from torch.nn import functional as F

from .layers import Concate_GCN
from dglt.contrib.moses.moses.data.reader.utils import mol_collate_func, weights_init
from deepchem.models.models import Model
from deepchem.trans import undo_transforms

class EAGCN(Model):
    def __init__(self, n_bfeat=41, n_afeat=25, n_sgc1_1=30, n_sgc1_2=10, n_sgc1_3=10, n_sgc1_4=10, n_sgc1_5=10,
                 n_sgc2_1=60, n_sgc2_2=20, n_sgc2_3=20, n_sgc2_4=20, n_sgc2_5=20,
                 n_den1=64, n_den2=32, attention_bidden=256, out_feature=128,
                 nclass=12, dropout=0.2):
        self.model = Concate_GCN(n_bfeat, n_afeat, n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                 n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5,
                 n_den1, n_den2, attention_bidden, out_feature,
                 nclass, dropout)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.apply(weights_init)

    def fit(self, dataset, learning_rate=5e-4, batch_size=32, epoch=10, **kwargs):
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=kwargs['weight_decay'])
        val_roc = []
        test_roc = []
        for i in range(epoch):
            for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(batch_size=batch_size):
                adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt = mol_collate_func(X_b)
                adj_batch, afm_batch, btf_batch = Variable(adj), Variable(afm), Variable(btf)
                orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(
                    aromAtt), Variable(conjAtt), Variable(ringAtt)
                label_batch = Variable(torch.from_numpy(y_b).float())
                w_b = Variable(torch.from_numpy(w_b).float())

                if torch.cuda.is_available():
                    label_batch = label_batch.cuda()
                    w_b = w_b.cuda()

                optimizer.zero_grad()
                outputs = self.model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch,
                                     ringAtt_batch)
                loss = F.binary_cross_entropy_with_logits(outputs, label_batch, w_b)
                loss.backward()
                optimizer.step()
            print('epoch '+ str(i) + ':')
            val_score = self.evaluate(kwargs['val_dataset'], kwargs['metric'], kwargs['transformer'])
            print(val_score)
            val_roc.append(val_score['mean-roc_auc_score'])
            test_score = self.evaluate(kwargs['test_dataset'], kwargs['metric'], kwargs['transformer'])
            print(test_score)
            test_roc.append(test_score['mean-roc_auc_score'])
        print('best test score refer to val:')
        print(test_roc[val_roc.index(max(val_roc))])


    def predict(self, dataset, transformers=[], batch_size=None):
        self.model.eval()
        y_preds = []
        y_true = []
        for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(batch_size=32, deterministic=True):
            adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt = mol_collate_func(X_b)
            adj_batch, afm_batch, btf_batch = Variable(adj), Variable(afm), Variable(btf)
            orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(
                aromAtt), Variable(conjAtt), Variable(ringAtt)
            label_batch = Variable(torch.from_numpy(y_b).float())
            w_b = Variable(torch.from_numpy(w_b).float())
            outputs = self.model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch,
                                 ringAtt_batch)
            outputs = F.sigmoid(outputs)
            outputs = outputs.data.cpu().numpy()
            outputs = np.stack([1 - outputs, outputs], -1)
            y_preds.append(outputs)
            y_true.append(y_b)
        return np.concatenate(y_preds)


