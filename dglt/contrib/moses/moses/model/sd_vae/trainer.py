import numpy as np
import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dglt.contrib.moses.moses.interfaces import MosesTrainer
from dglt.contrib.moses.moses.model.vae.misc import CosineAnnealingLRWithRestart, KLAnnealer
from dglt.contrib.moses.moses.model.sd_vae.utils import cfg_parser as parser
from dglt.contrib.moses.moses.utils import Logger, CircularBuffer
from .preprocess.dataset import SDVAEDataset
from .utils.mol_util import MolUtil


class SDVAETrainer(MosesTrainer):
    def __init__(self, config):
        self.config = config
        self.utils = MolUtil(config)
        self.grammar = None

    def get_vocabulary(self, data, extra=None):
        self.grammar = parser.Grammar(self.config.grammar_file)
        return self.grammar

    def get_collate_fn(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            onehot, masks, _ = zip(*data)
            if model.onehot:
                onehot = torch.tensor(np.stack(onehot), dtype=torch.float, device=device)
                masks = torch.tensor(np.stack(masks), dtype=torch.float, device=device)
            else:
                order = np.argsort([len(_) for _ in onehot])[::-1]
                onehot = [torch.tensor(onehot[_], dtype=torch.long, device=device) for _ in order]
                masks = [masks[_] for _ in order]
                masks = torch.tensor(np.stack(masks), dtype=torch.float, device=device)

            return onehot, masks

        return collate

    def _train_epoch(self, model, epoch, tqdm_data, kl_weight, device, optimizer=None):
        total_loss = CircularBuffer(self.config.n_last)
        recon_loss = CircularBuffer(self.config.n_last)
        kl_loss = CircularBuffer(self.config.n_last)

        if optimizer is None:
            model.eval()
        else:
            model.train()

        n_samples = 0
        for i, input_batch in enumerate(tqdm_data):
            if model.onehot:
                x_inputs = input_batch[0].clone().permute(0, 2, 1).to(device)
                v_tb = input_batch[0].permute(1, 0, 2).to(device)
                v_ms = input_batch[1].permute(1, 0, 2).to(device)

                # Forward
                loss_list = model(
                    x_inputs,
                    v_tb,
                    v_ms
                )
            else:
                x_inputs = tuple(data.to(device) for data in input_batch[0])
                v_ms = input_batch[1].to(device)

                # Forward
                loss_list = model(
                    x_inputs,
                    v_ms
                )

            perp = loss_list[0]

            if len(loss_list) == 1:  # only perplexity
                loss = loss_list[0]
                kl = 0
            else:
                loss = loss_list[0] + kl_weight * loss_list[1]
                kl = loss_list[1]

            minibatch_loss = loss.data

            # Backward
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.get_optim_params(model),
                                self.config.clip_grad)
                optimizer.step()

            # Log
            total_loss.add(minibatch_loss.item())
            recon_loss.add(perp.item())
            kl_loss.add(kl.item())
            lr = (optimizer.param_groups[0]['lr']
                  if optimizer is not None
                  else 0)

            # Update tqdm
            kl_loss_value = kl_loss.mean()
            recon_loss_value = recon_loss.mean()
            loss_value = total_loss.mean()
            postfix = [f'loss={loss_value:.5f}',
                       f'(kl={kl_loss_value:.5f}',
                       f'recon={recon_loss_value:.5f})',
                       f'klw={kl_weight:.5f},',
                       f'lr={lr:.5f}']
            tqdm_data.set_postfix_str(' '.join(postfix))

        postfix = {
            'epoch': epoch,
            'lr': lr,
            'kl_weight': kl_weight,
            'kl_loss': kl_loss_value,
            'recon_loss': recon_loss_value,
            'loss': loss_value,
            'mode': 'Eval' if optimizer is None else 'Train'}

        return postfix

    def get_optim_params(self, model):
        return (p for p in model.parameters() if p.requires_grad)

    def _train(self, model, train_loader, val_loader=None, logger=None):
        device = model.device
        n_epoch = self._n_epoch()

        optimizer = optim.Adam(self.get_optim_params(model),
                               lr=self.config.lr_start)
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=self.config.lr_factor,
                                         patience=self.config.lr_patience, verbose=True,
                                         min_lr=self.config.lr_end)
        kl_annealer = KLAnnealer(n_epoch, self.config)

        model.zero_grad()
        best_valid_loss = None
        for epoch in range(n_epoch):
            kl_weight = kl_annealer(epoch)
            tqdm_data = tqdm(train_loader,
                             desc='Training (epoch #{})'.format(epoch))
            postfix = self._train_epoch(model, epoch, tqdm_data, kl_weight, device, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader,
                                 desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, epoch, tqdm_data,
                                            (self.config.kl_w_start + self.config.kl_w_end) / 2,
                                            device)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            if self.config.model_save is not None:
                model = model.to('cpu')
                if epoch % self.config.save_frequency == 0:
                    torch.save(model.state_dict(),
                               self.config.model_save[:-3] +
                               '_{0:03d}.pt'.format(epoch))
                if best_valid_loss is None or postfix['loss'] < best_valid_loss:
                    best_valid_loss = postfix['loss']
                    print('----saving to best model since this is the best valid loss so far.----')
                    torch.save(model.state_dict(),
                               self.config.model_save[:-3] +
                               '_epoch-best.model')
                model = model.to(device)
            lr_scheduler.step(postfix['loss'])

    def fit(self, model, train_data, val_data=None, design=False):
        logger = Logger() if self.config.log_file is not None else None

        if 'SDVAE' in train_data.columns:
            train_data = train_data['SDVAE'].values
        else:
            train_data = train_data['SMILES'].values
        if val_data is not None:
            if 'SDVAE' in val_data.columns:
                val_data = val_data['SDVAE'].values
            else:
                val_data = val_data['SMILES'].values

        train_loader = self.get_dataloader(model, SDVAEDataset(train_data, self.config,
                                                               grammar=self.grammar,
                                                               onehot=model.onehot),
                                           shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(
            model, SDVAEDataset(val_data, self.config, grammar=self.grammar, onehot=model.onehot),
            shuffle=False
        )

        self._train(model, train_loader, val_loader, logger)
        return model

    def _n_epoch(self):
        return sum(
            self.config.lr_n_period * (self.config.lr_n_mult ** i)
            for i in range(self.config.lr_n_restarts)
        )

