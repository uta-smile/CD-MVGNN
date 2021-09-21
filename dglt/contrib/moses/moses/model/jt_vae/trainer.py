import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

import math, random, sys
import numpy as np
import argparse
from collections import deque
#import cPickle as pickle
import pickle as pickle
from tqdm import tqdm
import rdkit


from dglt.contrib.moses.moses.interfaces import MosesTrainer
from dglt.contrib.moses.moses.utils import OneHotVocab, Logger, CircularBuffer
from dglt.contrib.moses.moses.model.vae.misc import CosineAnnealingLRWithRestart, KLAnnealer

from dglt.contrib.moses.moses.model.jt_vae.mol_tree import Vocab, MolTree
from dglt.contrib.moses.moses.model.jt_vae.nnutils import create_var
from dglt.contrib.moses.moses.model.jt_vae.datautils import MolTreeFolder, PairTreeFolder, MolTreeDataset

class JTVAETrainer(MosesTrainer):
    def __init__(self, config):
        self.config = config

    def get_vocabulary(self, data):
        #return OneHotVocab.from_data(data)
        vocab = [x.strip("\r\n ") for x in open(self.config.vocab)] 
        vocab = Vocab(vocab)
        
        return vocab

    def get_collate_fn(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device=device)
                       for string in data]

            return tensors

        return collate

    def _train_epoch(self, model, epoch, tqdm_data, kl_weight, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        kl_loss_values = CircularBuffer(self.config.n_last)
        recon_loss_values = CircularBuffer(self.config.n_last)
        loss_values = CircularBuffer(self.config.n_last)
        for i, input_batch in enumerate(tqdm_data):
            input_batch = tuple(data.to(model.device) for data in input_batch)

            # Forward
            kl_loss, recon_loss = model(input_batch)
            loss = kl_weight * kl_loss + recon_loss

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
            loss_values.add(loss.item())
            lr = (optimizer.param_groups[0]['lr']
                  if optimizer is not None
                  else None)

            # Update tqdm
            kl_loss_value = kl_loss_values.mean()
            recon_loss_value = recon_loss_values.mean()
            loss_value = loss_values.mean()
            postfix = [f'loss={loss_value:.5f}',
                       f'(kl={kl_loss_value:.5f}',
                       f'recon={recon_loss_value:.5f})',
                       f'klw={kl_weight:.5f} lr={lr:.5f}']
            tqdm_data.set_postfix_str(' '.join(postfix))

        postfix = {
            'epoch': epoch,
            'kl_weight': kl_weight,
            'lr': lr,
            'kl_loss': kl_loss_value,
            'recon_loss': recon_loss_value,
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
        kl_annealer = KLAnnealer(n_epoch, self.config)
        lr_annealer = CosineAnnealingLRWithRestart(optimizer,
                                                   self.config)

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
                postfix = self._train_epoch(model, epoch, tqdm_data, kl_weight)
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
            lr_annealer.step()

    def fit(self, model, train_data, val_data=None):

        lg = rdkit.RDLogger.logger() 
        lg.setLevel(rdkit.RDLogger.CRITICAL)


        args = self.config
        print(args)

        vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
        vocab = Vocab(vocab)

        print(args.latent_size)
        model = model.cuda()
        print(model)

        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

        if args.load_epoch > 0:
            model.load_state_dict(torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch)))

        print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
        scheduler.step()

        param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
        grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

        total_step = args.load_epoch
        beta = args.beta
        meters = np.zeros(4)

        for epoch in range(args.epoch):
            loader = MolTreeFolder(args.train_prep, vocab, args.batch_size, num_workers=4)
            #loader = MolTreeFolderLabel(args.train_prep, vocab, args.batch_size, args.property, num_workers=4)
            for batch in loader:
                total_step += 1
                try:
                    model.zero_grad()
                    #loss, kl_div, mse, wacc, tacc, sacc = model(batch, beta)
                    loss, kl_div, wacc, tacc, sacc = model(batch, beta)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                    optimizer.step()
                except Exception as e:
                    print(e)
                    continue

                #meters = meters + np.array([kl_div, mse, wacc * 100, tacc * 100, sacc * 100])
                meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

                if total_step % args.print_iter == 0:
                    meters /= args.print_iter
                    #print("[%d] Beta: %.3f, KL: %.2f, MSE: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], param_norm(model), grad_norm(model)))
                    print("[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model)))
                    sys.stdout.flush()
                    meters *= 0

                if total_step % args.save_iter == 0:
                    torch.save(model.state_dict(), args.train_save_dir + "/model.iter-" + str(total_step))

                if total_step % args.anneal_iter == 0:
                    scheduler.step()
                    print("learning rate: %.6f" % scheduler.get_lr()[0])

                if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
                    beta = min(args.max_beta, beta + args.step_beta)


        return model

    def _n_epoch(self):
        return sum(
            self.config.lr_n_period * (self.config.lr_n_mult ** i)
            for i in range(self.config.lr_n_restarts)
        )


