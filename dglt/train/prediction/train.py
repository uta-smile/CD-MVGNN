import logging
from argparse import Namespace
from typing import Callable, List, Union

import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from dglt.data.dataset.molecular import MoleculeDataset
from dglt.data.transformer.collator import MolCollator
from dglt.models.nn_utils import compute_gnorm, compute_pnorm, NoamLR
from dglt.train.prediction.utils import heartbeat


def train(epoch,
          model: nn.Module,
          data: Union[MoleculeDataset, List[MoleculeDataset]],
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          shared_dict,
          args: Namespace,
          n_iter: int = 0,
          logger: logging.Logger = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print
    
    model.train()
    
    # data.shuffle()

    loss_sum, iter_count = 0, 0
    cum_loss_sum, cum_iter_count = 0, 0

    num_iters = len(data) // args.batch_size * args.batch_size  # don't use the last batch if it's small, for stability

    iter_size = args.batch_size

    mol_collator = MolCollator(shared_dict=shared_dict, args=args)
    # mol_dataset = MoleculeDataset(data)
    if args.enbl_multi_gpu:
        from dglt.multi_gpu_wrapper import MultiGpuWrapper as mgw

        # distributed version: use DistributedSampler to partition data among workers. Manually specify
        # `num_replicas=mgw.size()` and `rank=mgw.rank()`.
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            data, num_replicas=mgw.size(), rank=mgw.rank())
        mol_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                                sampler=train_sampler, num_workers=4, collate_fn=mol_collator)

        # Horovod: set epoch to sampler for shuffling
        train_sampler.set_epoch(epoch)
    else:
        if args.debug:
            num_workers = 0
        else:
            num_workers = 4
        mol_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                                num_workers=num_workers, collate_fn=mol_collator)


    for i, item in enumerate(mol_loader):
        smiles_batch, batch, features_batch, mask, targets = item
        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()
        class_weights = torch.ones(targets.shape)

        if args.cuda:
            class_weights = class_weights.cuda()

        # Run model
        model.zero_grad()
        preds = model(batch, features_batch)
        loss = loss_func(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += args.batch_size

        cum_loss_sum += loss.item()
        cum_iter_count += 1

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += args.batch_size

        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0
            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
        if os.environ.get('CHIEF_IP', ''):
            heartbeat()

    return n_iter, cum_loss_sum / cum_iter_count




