import os
import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import ReduceLROnPlateau

# hyper-parameters
BATCH_SIZE = 64
NB_EPOCHS = 50
NB_FEAT_DIMS = 56  #112
NB_LABL_DIMS = 5
DATA_DIR = '/data1/jonathan/Molecule.Generation/AIPharmacist-data'
NPZ_PATH_TRN = os.path.join(DATA_DIR, 'train_w_props.npz')
NPZ_PATH_TST = os.path.join(DATA_DIR, 'test_w_props.npz')
MODEL_DIR = '/data1/jonathan/Molecule.Generation/AIPharmacist-models'
MODEL_PATH = os.path.join(MODEL_DIR, 'gvae_props.model')

class Model(nn.Module):
    """Molecule property estimator - Model."""

    def __init__(self):
        """Constructor function."""

        super().__init__()
        self.model = self.__build_model()

    def forward(self, x):
        return self.model(x)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def params(self):
        return (p for p in self.parameters() if p.requires_grad)

    def __build_model(self):
        """Build a model."""

        model = nn.Sequential(
            nn.Linear(NB_FEAT_DIMS, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(32, NB_LABL_DIMS)
        )

        return model

class Trainer(object):
    """Molecule property estimator - Trainer."""

    def __init__(self, model, npz_path_trn, npz_path_tst):
        """Constructor function."""

        self.model = model
        self.val_split = 0.1
        self.lr_init = 1e-3
        self.lr_min = 1e-5

        self.__create_data_loader(npz_path_trn, is_train=True)
        self.__create_data_loader(npz_path_tst, is_train=False)

    def run(self):
        """Train a MLP model for property estimation, and test it periodically."""

        self.model.to('cuda')
        optimizer = torch.optim.Adam(self.model.params, lr=self.lr_init)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=self.lr_min)
        for idx_epoch in range(NB_EPOCHS):
            # train
            loss_trn = 0.0
            self.model.train()
            for batch in tqdm(self.data_loader_trn, desc='Train (epoch #%d)' % (idx_epoch + 1)):
                optimizer.zero_grad()
                preds = self.model.forward(batch[0])
                loss = torch.nn.MSELoss()(preds, batch[1])
                loss_trn += loss.item()
                loss.backward()
                optimizer.step()
            loss_trn /= len(self.data_loader_trn)
            lr = optimizer.param_groups[0]['lr']

            # validation
            loss_val = 0.0
            self.model.eval()
            for batch in tqdm(self.data_loader_val, desc='Valid (epoch #%d)' % (idx_epoch + 1)):
                preds = self.model.forward(batch[0])
                loss = torch.nn.MSELoss()(preds, batch[1])
                loss_val += loss.item()
            loss_val /= len(self.data_loader_val)
            lr_scheduler.step(loss_val)

            # test
            loss_tst = 0.0
            for batch in tqdm(self.data_loader_tst, desc='Test (epoch #%d)' % (idx_epoch + 1)):
                preds = self.model.forward(batch[0])
                loss = torch.nn.MSELoss()(preds, batch[1])
                loss_tst += loss.item()
            loss_tst /= len(self.data_loader_tst)

            print('Epoch #%d: lr = %e | loss = %e (trn) / %e (val) / %e (tst)'
                  % (idx_epoch + 1, lr, loss_trn, loss_val, loss_tst))

    def run_eval(self):
        """Evaluate the MLP model on validation & test subsets."""

        self.model.to('cuda')

        # validation
        loss_val = 0.0
        self.model.eval()
        for batch in tqdm(self.data_loader_val, desc='[Eval] Valid'):
            preds = self.model.forward(batch[0])
            loss = torch.nn.MSELoss()(preds, batch[1])
            loss_val += loss.item()
        loss_val /= len(self.data_loader_val)

        # test
        loss_tst = 0.0
        for batch in tqdm(self.data_loader_tst, desc='[Eval] Test'):
            preds = self.model.forward(batch[0])
            loss = torch.nn.MSELoss()(preds, batch[1])
            loss_tst += loss.item()
        loss_tst /= len(self.data_loader_tst)

        print('[Eval] loss = %e (val) / %e (tst)' % (loss_val, loss_tst))

    def __create_data_loader(self, npz_path, is_train):
        """Create a data loader from *.npz file."""

        def __create_from_numpy(x, y):
            x_tns = torch.from_numpy(x).to('cuda')
            y_tns = torch.from_numpy(y).to('cuda')
            dataset = TensorDataset(x_tns, y_tns)
            data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            return data_loader

        npz_data = np.load(npz_path)
        feats, labls = npz_data['feats'], npz_data['labls']
        feats = feats[:, :NB_FEAT_DIMS]
        print('features: {} / {}'.format(feats.shape, feats.dtype))
        print('labels  : {} / {}'.format(labls.shape, labls.dtype))

        if is_train:
            idxs = np.arange(feats.shape[0])
            np.random.shuffle(idxs)
            feats, labls = feats[idxs], labls[idxs]
            nb_smpls_trn = int(feats.shape[0] * (1.0 - self.val_split))
            self.data_loader_trn = __create_from_numpy(feats[:nb_smpls_trn], labls[:nb_smpls_trn])
            self.data_loader_val = __create_from_numpy(feats[nb_smpls_trn:], labls[nb_smpls_trn:])
        else:
            self.data_loader_tst = __create_from_numpy(feats, labls)

class FeatOptimizer(object):
    """Molecule property estimator - Feature optimizer."""

    def __init__(self, model):
        """Constructor function."""

        self.model = model
        self.lr_init = 1e-1
        self.momentum = 0.9
        self.lr_min = 1e-3
        self.nb_iters = 256

    def run(self, feats_init, labl_values, labl_coeffs):

        self.model.to('cuda')
        self.model.eval()
        feats = feats_init.to('cuda')
        feats.requires_grad = True
        labl_values = labl_values.to('cuda')
        labl_coeffs = labl_coeffs.to('cuda')

        optimizer = torch.optim.Adam([feats], lr=self.lr_init)
        #optimizer = torch.optim.SGD([feats], lr=self.lr_init, momentum=self.momentum)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=self.lr_min)
        for idx_iter in range(self.nb_iters):
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            preds = self.model(feats)
            loss = torch.mean(labl_coeffs * (preds - labl_values) ** 2)
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)
            #print('iter #%d: lr = %e | loss = %e' % (idx_iter + 1, lr, loss))

        feats_finl = feats.detach()
        preds_finl = preds.detach()

        return feats_finl, preds_finl

def main():
    """Main entry."""

    # train a model from scratch, or restore a pre-trained model
    model = Model()
    trainer = Trainer(model, NPZ_PATH_TRN, NPZ_PATH_TST)
    if not os.path.exists(MODEL_PATH):
        trainer.run()
        model.to('cpu')
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        model_state = torch.load(MODEL_PATH)
        model.load_state_dict(model_state)
        trainer.run_eval()

    # test the feature optimizer
    feats_init = Normal(0.0, 1.0).sample((BATCH_SIZE, NB_FEAT_DIMS)) * 1e-1
    labl_values = torch.tensor([[0.7, 0.2, 0.3, 0.4, 0.2]]).repeat(BATCH_SIZE, 1)
    labl_coeffs = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]).repeat(BATCH_SIZE, 1)
    feat_optimizer = FeatOptimizer(model)
    feats_finl, preds_finl = feat_optimizer.run(feats_init, labl_values, labl_coeffs)

    #preds_finl_np = preds_finl.to('cpu').numpy()
    #for idx in range(BATCH_SIZE):
    #    print(' | '.join(['%7.4f' % preds_finl_np[idx][ic] for ic in range(NB_LABL_DIMS)]))

if __name__ == '__main__':
    main()
