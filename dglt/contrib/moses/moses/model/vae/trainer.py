import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from dglt.contrib.moses.moses.interfaces import MosesTrainer
from dglt.contrib.moses.moses.utils import OneHotVocab, Logger, CircularBuffer
from dglt.contrib.moses.moses.model.vae.misc import CosineAnnealingLRWithRestart, KLAnnealer
from dglt.contrib.moses.moses.model.vae.dataset import VAEDataset
from dglt.multi_gpu_wrapper import MultiGpuWrapper as mgw


class VAETrainer(MosesTrainer):
    def __init__(self, config):
        self.config = config

    def get_vocabulary(self, data, extra=None):
        return OneHotVocab.from_data(list(data['SMILES']), extra)

    def get_collate_fn(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            data = np.stack(data)
            order = np.argsort(list(map(len, data[:, 0])))[::-1]

            x = [model.string2tensor(string, device=device)
                       for string in data[order, 0]]
            y = torch.tensor(data[order,1:].astype(float), dtype=torch.float, device=device)

            return x, y

        return collate

    def _train_epoch(self, model, epoch, tqdm_data, kl_weight, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        kl_loss_values = CircularBuffer(self.config.n_last)
        recon_loss_values = CircularBuffer(self.config.n_last)
        mse_loss_values = CircularBuffer(self.config.n_last)
        loss_values = CircularBuffer(self.config.n_last)
        for i, (input_batch, labels) in enumerate(tqdm_data):
            input_batch = tuple(data.to(model.device) for data in input_batch)
            labels = labels.to(model.device)

            # Forward
            kl_loss, recon_loss, mse_loss = model(input_batch, labels)
            loss = kl_weight * (kl_loss + self.config.mse_weight * mse_loss) + recon_loss

            # Backward
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.get_optim_params(model),
                                self.config.clip_grad)
                optimizer.step()

            # Log
            kl_loss_values.add(kl_loss.item())
            recon_loss_values.add(recon_loss.item())
            mse_loss_values.add(mse_loss.item())
            loss_values.add(loss.item())
            lr = (optimizer.param_groups[0]['lr']
                  if optimizer is not None else 0.0)

            # Update tqdm
            kl_loss_value = kl_loss_values.mean()
            recon_loss_value = recon_loss_values.mean()
            mse_loss_value = mse_loss_values.mean()
            loss_value = loss_values.mean()
            postfix = [f'loss={loss_value:.5f}',
                       f'(kl={kl_loss_value:.5f}',
                       f'recon={recon_loss_value:.5f}',
                       f'mse={mse_loss_value:.5f})',
                       f'klw={kl_weight:.5f}',
                       f'msew={self.config.mse_weight:.5f}',
                       f'lr={lr:.5f}']
            tqdm_data.set_postfix_str(' '.join(postfix))

        postfix = {
            'epoch': epoch,
            'kl_weight': kl_weight,
            'mse_weight': self.config.mse_weight,
            'lr': lr,
            'kl_loss': kl_loss_value,
            'recon_loss': recon_loss_value,
            'mse_loss': mse_loss_value,
            'loss': loss_value,
            'mode': 'Eval' if optimizer is None else 'Train'}

        return postfix

    def get_optim_params(self, model):
        return (p for p in model.vae.parameters() if p.requires_grad)

    def _train(self, model, train_loader, val_loader=None, logger=None):
        device = model.device
        n_epoch = self._n_epoch()

        optimizer = optim.Adam(self.get_optim_params(model),
                               lr=self.config.lr_start)

        if self.config.enable_multi_gpu:
            mgw.broadcast_parameters(model.state_dict(), root_rank=0)
            optimizer = mgw.DistributedOptimizer(optimizer,
                                                 named_parameters=model.named_parameters())

        kl_annealer = KLAnnealer(n_epoch, self.config)
        lr_annealer = ReduceLROnPlateau(optimizer, 'min', factor=self.config.lr_factor,
                                         patience=self.config.lr_patience, verbose=True,
                                         min_lr=self.config.lr_end)

        model.zero_grad()
        for epoch in range(n_epoch):
            # Epoch start
            kl_weight = kl_annealer(epoch)

            tqdm_data = tqdm(train_loader,
                             desc='Training (epoch #{})'.format(epoch))
            postfix = self._train_epoch(model, epoch,
                                        tqdm_data, kl_weight, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader,
                                 desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, epoch, tqdm_data, self.config.kl_w_end)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            if (self.config.model_save is not None) and \
                    (epoch % self.config.save_frequency == 0):
                model = model.to('cpu')
                torch.save(model.state_dict(),
                           self.config.model_save[:-3] +
                           '_{0:03d}.pt'.format(epoch))
                model = model.to(device)

            # Epoch end
            lr_annealer.step(postfix['loss'])

    def fit(self, model, train_data, val_data=None):
        logger = Logger() if self.config.log_file is not None else None

        # Model fine tune
        if self.config.model_load is not None and 'fix_emb' in self.config.fine_tune:
            for layer in model.x_emb.parameters():
                layer.requires_grad = False
        if self.config.model_load is not None and 'fix_enc' in self.config.fine_tune:
            for layer in model.encoder.parameters():
                layer.requires_grad = False

        # Load data with/without parallel
        train_data = VAEDataset(train_data, self.config)
        is_parallel = isinstance(model, torch.nn.DataParallel)
        if is_parallel:
            if model.module.load_norm:
                train_data.update_norm(model.module.min.data.cpu().numpy(),
                                       model.module.max.data.cpu().numpy())
            else:
                model.module.update_norm(train_data.min, train_data.max)
        else:
            if model.load_norm:
                train_data.update_norm(model.min.data.cpu().numpy(),
                                       model.max.data.cpu().numpy())
            else:
                model.update_norm(train_data.min, train_data.max)

        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(
            model, VAEDataset(val_data, self.config), shuffle=False
        )

        self._train(model, train_loader, val_loader, logger)
        return model

    def _n_epoch(self):
        return self.config.n_epoch
