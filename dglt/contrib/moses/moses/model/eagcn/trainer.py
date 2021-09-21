import os
import sys


import numpy as np
import deepchem as dc
from dglt.contrib.moses.moses.data.reader.tox21_datasets import load_tox21, tox21_ad
from dglt.contrib.moses.moses.data.reader.bace_datasets import load_bace_classification, bace_ad
from dglt.contrib.moses.moses.data.reader.clintox_datasets import load_clintox, clintox_ad
from dglt.contrib.moses.moses.model.eagcn import EAGCN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


tasks, datasets, transformers = load_clintox(featurizer='EAGCN', split='scaffold')
dict_len = clintox_ad.dict_len
train_dataset, valid_dataset, test_dataset = datasets

metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

dropout = 0.2
n_den1 = 64
n_den2 = 32
attention_hidden = 256
out_feature = 128

lr = 5e-4
batch_size = 32
weight_decay = 4e-5
epoch = 100

model = EAGCN(dropout=dropout, n_den1=n_den1, n_den2=n_den2, attention_bidden=attention_hidden,
              out_feature=out_feature, n_bfeat=dict_len, nclass=len(tasks))

model.fit(train_dataset, learning_rate=lr, batch_size=batch_size, weight_decay=weight_decay, epoch=epoch,
          val_dataset=valid_dataset, test_dataset=test_dataset, transformer=transformers, metric= [metric])

