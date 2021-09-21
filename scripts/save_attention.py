import pickle
from argparse import Namespace, ArgumentParser
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dglt.data.dataset.molecular import MoleculeDataset
from dglt.data.dataset.utils import get_task_names, get_data
from dglt.data.transformer.collator import MolCollator
from dglt.utils import load_checkpoint


def add_args(parser):
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to the label of fg task.')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--checkpoint_paths', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')

    parser.add_argument('--hidden_size', type=int, default=3,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to linear layers')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--ffn_hidden_size', type=int, default=None,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')
    parser.add_argument('--undirected', action='store_true', default=False,
                        help='Undirected edges (always sum the two relevant bond vectors)')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')

    parser.add_argument('--dense', action='store_true', default=False,
                        help='Cat message in different time stamp in mpn')

    # self-attention readout.
    parser.add_argument('--self_attention', action='store_true', default=False, help='Use self attention layer')
    parser.add_argument('--attn_hidden', type=int, default=4, nargs='?', help='self attention layer hidden layer size')
    parser.add_argument('--attn_out', type=int, default=128, nargs='?', help='self attention layer output feature size')

    # Arguments for DualMPNN
    parser.add_argument('--dist_coff', type=float, default=0.1, help='The dist coefficient for the DualMPNN.')

    parser.add_argument('--distinct_init', action='store_true', default=False,
                        help='Using distinct weight init for model ensemble')


def predict(model: nn.Module,
            data: MoleculeDataset,
            args,
            batch_size: int,
            logger=None,
            ):
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    attns = []
    smiles = []
    targetss = []
    shared_dict = {}
    debug = logger.debug if logger is not None else print
    model.eval()
    args.bond_drop_rate = 0
    preds = []

    num_iters, iter_step = len(data), batch_size
    loss_sum, iter_count = 0, 0

    mol_collator = MolCollator(args=args, shared_dict=shared_dict)
    if args.debug:
        num_workers = 0
    else:
        num_workers = 4
    mol_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=mol_collator)

    for i, item in enumerate(mol_loader):
        smiles_batch, batch, features_batch, mask, targets = item
        class_weights = torch.ones(targets.shape)
        if next(model.parameters()).is_cuda:
            targets = targets.cuda()
        with torch.no_grad():
            batch_preds = model(batch, features_batch)
            attns.extend(model.encoders.readout.attns)
            smiles.extend(smiles_batch)
            targetss.extend(targets.cpu().numpy())

            iter_count += 1
        # Collect vectors
        batch_preds = batch_preds.data.cpu().numpy().tolist()
        preds.extend(batch_preds)

    return attns, smiles, targetss


def run_training(args: Namespace) -> List[float]:
    debug = info = print
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path,
                    args=args, logger=None)
    train_data = data
    args.train_data_size = len(train_data)
    model = load_checkpoint(args.checkpoint_paths, current_args=args, logger=None)
    debug(model)
    model = model.cuda()
    attns, smiles, targets = predict(model,
                                     train_data,
                                     args,
                                     args.batch_size,
                                     logger=None,
                                     )
    with open(args.save_path, "wb") as f:
        pickle.dump((attns, smiles, targets), f)


def run():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    args.max_data_size = None
    args.use_compound_names = False
    args.features_generator = None
    args.aug_rate = 0
    args.cuda = True
    args.num_tasks = 1
    args.model_type = "dualmpnn"
    args.coord = False
    args.atom_messages = False
    args.use_input_features = True
    args.features_only = False
    args.no_attach_fea = True
    args.dataset_type = 'classification'
    args.input_layer = "fc"
    args.debug = True
    args.no_cache = True
    run_training(args)


if __name__ == '__main__':
    run()
