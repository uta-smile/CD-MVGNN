import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

from torch.nn.utils import clip_grad_norm_

from dglt.contrib.moses.moses.interfaces import MosesTrainer
from dglt.contrib.moses.moses.utils import OneHotVocab, Logger, CircularBuffer
from dglt.contrib.moses.moses.model.gvae.misc import KLAnnealer
from dglt.contrib.moses.moses.model.mpnn.dataset import MPGVAEDataset
from dglt.contrib.moses.moses.data.mpnn.data.utils import MolCollator


class MPGVAETrainer(MosesTrainer):
    def __init__(self, config):
        self.config = config
        self.mol_collator = MolCollator({}, self.config)

    def fit(self, model, train_data, val_data=None):
        """Update model parameters to fit training samples."""

        # three subsets are involved:
        # - trn-trn: training subset of <train_data> to update model parameters
        # - trn-val: validation subset of <train_data> to update the learning rate
        # - val:     <val_data> (may not be provided)

        # configure the logging module
        self.logger = Logger() if self.config.log_file is not None else None

        # Model fine tune
        if self.config.model_load is not None and 'fix_emb' in self.config.fine_tune:
            for layer in model.x_emb.parameters():
                layer.requires_grad = False
        if self.config.model_load is not None and 'fix_enc' in self.config.fine_tune:
            for layer in model.encoder_rnn.parameters():
                layer.requires_grad = False

        # get GVAE column
        # train_data = np.squeeze(train_data['SMILES'].values).tolist()
        # val_data = None if val_data is None else np.squeeze(val_data['SMILES'].values).tolist()

        # split <train_data> into trn-trn & trn-val subsets
        n_samples = len(train_data)
        n_tval_samples = int(n_samples * self.config.r_tval_samples)
        print('using %d out of %d samples to update LR' % (n_tval_samples, n_samples))
        ttrn_loader = self.get_dataloader(
            model,
            MPGVAEDataset(train_data[n_tval_samples:], self.config),
            shuffle=True
        )
        tval_loader = self.get_dataloader(
            model,
            MPGVAEDataset(train_data[:n_tval_samples], self.config),
            shuffle=True
        )

        # build a data loader for validation samples, if needed
        val_loader = None
        if val_data is not None:
            val_loader = self.get_dataloader(model, MPGVAEDataset(val_data, self.config),
                                             shuffle=False)

        # call the model training routine
        self.__train(model, ttrn_loader, tval_loader, val_loader)

        return model

    def get_vocabulary(self, data, extra=None):
        """Get the vocabulary."""

        with open(self.config.rule_path, 'r') as i_file:
            vocab = [i_line.strip() for i_line in i_file]

        return vocab

    def get_collate_fn(self, model):
        """Get the data pre-processing function."""

        device = self.get_collate_device(model)
        def collate_fn(data):
            mpnn_dataset, gvae_dataset = zip(*data)
            batch, feature_batch, _, _ = self.mol_collator(mpnn_dataset)
            gvae_data = [model.raw2tensor(x, device=device) for x in gvae_dataset]
            return batch, feature_batch, gvae_data

        return collate_fn

    def __train(self, model, ttrn_loader, tval_loader, val_loader=None):

        device = model.device
        n_epoch = self.config.n_epoch

        model.zero_grad()
        optimizer = optim.Adam(model.optim_params, lr=self.config.lr_init)
        kl_annealer = KLAnnealer(n_epoch, self.config)
        lr_scheduler = ReduceLROnPlateau(
            optimizer, factor=self.config.lr_factor, patience=3, min_lr=self.config.lr_min)
        for epoch in range(n_epoch):
            print('starting the %d-th epoch (%d in total)' % (epoch + 1, n_epoch))

            # obtain the KL-divergence loss's coefficient
            kl_coeff = kl_annealer(epoch + 1)

            # train the model on the <trn-trn> subset
            tqdm_data = tqdm(ttrn_loader, desc='Trn-Trn (epoch #{})'.format(epoch))
            self._train_epoch(model, epoch, tqdm_data, kl_coeff, optimizer)

            # evaluate the model on the <trn-val> subset
            tqdm_data = tqdm(tval_loader, desc='Trn-Val (epoch #{})'.format(epoch))
            postfix = self._train_epoch(model, epoch, tqdm_data, kl_coeff)
            lr_scheduler.step(postfix['loss'])

            # evaluate the model on the <val> subset
            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc='Validation (epoch #{})'.format(epoch))
                self._train_epoch(model, epoch, tqdm_data, kl_coeff)

            # (periodically) save the model
            if self.config.model_save and (epoch + 1) % self.config.save_frequency == 0:
                model = model.to('cpu')
                model_path = self.config.model_save[:-3] + '_{0:03d}.pt'.format(epoch)
                torch.save(model.state_dict(), model_path)
                model = model.to(device)

    def _train_epoch(self, model, epoch, tqdm_data, kl_coeff, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        kl_loss_values = CircularBuffer(self.config.n_last)
        recon_loss_values = CircularBuffer(self.config.n_last)
        loss_values = CircularBuffer(self.config.n_last)
        acc_all_values = CircularBuffer(self.config.n_last)
        acc_vld_values = CircularBuffer(self.config.n_last)
        acc_pad_values = CircularBuffer(self.config.n_last)
        for i, input_batch in enumerate(tqdm_data):
            # group samples into a mini-batch
            #input_batch = torch.cat([torch.unsqueeze(x, dim=0) for x in input_batch], dim=0)
            batch, feature_batch, gvae_batch = input_batch
            gvae_batch = tuple(x.to(model.device) for x in gvae_batch)

            # Forward
            kl_loss, recon_loss, acc_all, acc_vld, acc_pad = model(batch, feature_batch, gvae_batch)
            loss = kl_coeff * kl_loss + recon_loss

            # Backward
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.optim_params, self.config.clip_grad)
                optimizer.step()

            # Log
            kl_loss_values.add(kl_loss.item())
            recon_loss_values.add(recon_loss.item())
            loss_values.add(loss.item())
            acc_all_values.add(acc_all.item())
            acc_vld_values.add(acc_vld.item())
            acc_pad_values.add(acc_pad.item())
            lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 0.0

            # Update tqdm
            kl_loss_value = kl_loss_values.mean()
            recon_loss_value = recon_loss_values.mean()
            loss_value = loss_values.mean()
            acc_all_value = acc_all_values.mean()
            acc_vld_value = acc_vld_values.mean()
            acc_pad_value = acc_pad_values.mean()
            postfix = [f'loss={loss_value:.5f}',
                       f'(kl={kl_loss_value:.5f}',
                       f'recon={recon_loss_value:.5f})',
                       f'acc_all={acc_all_value:.5f}',
                       f'acc_vld={acc_vld_value:.5f}',
                       f'acc_pad={acc_pad_value:.5f}',
                       f'lr={lr:.5f}']
            tqdm_data.set_postfix_str(' '.join(postfix))

        postfix = {
            'epoch': epoch,
            'lr': lr,
            'kl_loss': kl_loss_value,
            'recon_loss': recon_loss_value,
            'loss': loss_value,
            'acc_all': acc_all_value,
            'acc_vld': acc_vld_value,
            'acc_pad': acc_pad_value,
            'mode': 'Eval' if optimizer is None else 'Train'}
        self.__append_string_to_log(postfix)

        return postfix

    def __append_string_to_log(self, string):
        """Append a string to the logging file."""

        if self.logger is not None:
            self.logger.append(string)
            self.logger.save(self.config.log_file)
