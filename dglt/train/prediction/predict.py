from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dglt.data.dataset.molecular import MoleculeDataset
from dglt.data.transformer.collator import MolCollator
from dglt.data.transformer.scaler import StandardScaler


def predict(model: nn.Module,
            data: MoleculeDataset,args,
            batch_size: int,
            loss_func,
            logger,
            shared_dict,
            scaler: StandardScaler = None
            ) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    debug = logger.debug if logger is not None else print
    model.eval()
    args.bond_drop_rate = 0
    preds = []

    num_iters, iter_step = len(data), batch_size
    loss_sum, iter_count = 0, 0

    mol_collator = MolCollator(args=args, shared_dict=shared_dict)
    # mol_dataset = MoleculeDataset(data)
    if args.debug:
        num_workers = 0
    else:
        num_workers = 4
    mol_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=mol_collator)
    for i, item in enumerate(mol_loader):
        smiles_batch, batch, features_batch, mask, targets = item
        class_weights = torch.ones(targets.shape)
        if next(model.parameters()).is_cuda:
            targets = targets.cuda()
            mask = mask.cuda()
            class_weights = class_weights.cuda()
        with torch.no_grad():
            batch_preds = model(batch, features_batch)
            if loss_func is not None:
                loss = loss_func(batch_preds, targets) * class_weights * mask
                loss = loss.sum() / mask.sum()
                loss_sum += loss.item()
            iter_count += 1
        # Collect vectors
        batch_preds = batch_preds.data.cpu().numpy().tolist()
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)
        preds.extend(batch_preds)

    loss_avg = loss_sum / iter_count
    return preds, loss_avg

    # for i in trange(0, num_iters, iter_step):
    #     # Prepare batch
    #     mol_batch = MoleculeDataset(data[i:i + batch_size])
    #     smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()
    #
    #     # Run model
    #     batch = smiles_batch
    #
    #     with torch.no_grad():
    #         batch_preds = model(batch, features_batch)
    #
    #     batch_preds = batch_preds.data.cpu().numpy()
    #
    #     # Inverse scale if regression
    #     if scaler is not None:
    #         batch_preds = scaler.inverse_transform(batch_preds)
    #
    #     # Collect vectors
    #     batch_preds = batch_preds.tolist()
    #     preds.extend(batch_preds)
    #
    # return preds
