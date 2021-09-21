import os
from argparse import ArgumentParser

import numpy as np
import torch

from dglt.contrib.grover.molberttrainer import run_training
from dglt.utils import create_logger

from dglt.contrib.grover.vocab import AtomVocab

def add_args(parser):
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--enable_multi_gpu', dest='enable_multi_gpu',
                        action='store_true', default=False,
                        help='enable multi-GPU training')

    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--fg_label_path', type=str, nargs='*',
                        help='Path to the label of fg task.')
    parser.add_argument('--vocab_path', type=str, help="Path to the vocabulary.")
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined'],
                        help='Method of splitting the data into train/val/test')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use when splitting data into train/val/test sets.'
                             'When `num_folds` > 1, the first fold uses this seed and all'
                             'subsequent folds add 1 to the seed.')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')



    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=10,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1.0,
                        help='Final learning rate')

    parser.add_argument('--hidden_size', type=int, default=3,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to linear layers')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='SELU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    parser.add_argument('--undirected', action='store_true', default=False,
                        help='Undirected edges (always sum the two relevant bond vectors)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')
    parser.add_argument('--select_by_loss', action='store_true', default=False,
                        help='Use validation loss as refence standard to select best model to predict')
    parser.add_argument('--skip_epoch', type=int, default=0, help='Start select the model after skip_epoch '
                                                                  'in training. Default is 0.')

    parser.add_argument('--aug_rate', type=float, default=0, help='On the fly smiles enumeration')
    parser.add_argument('--no_attach_fea', action='store_true', default=False,
                        help="Do not attach feature in message passing.")

    parser.add_argument('--dist_coff', type=float, default=0.1, help='The dist coefficient for the DualMPNN.')
    parser.add_argument('--bond_drop_rate', type=float, default=0, help='Drop out bond in molecular')
    parser.add_argument('--save_interval', type=int, default=9999999999, help='The model saving interval.')

    parser.add_argument('--input_layer', type=str, default='fc',
                        choices=['fc', 'gcn'],
                        help='The input layer of model')
def train():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    # Compatibility issue
    args.features_path = args.fg_label_path
    args.max_data_size = float('inf')
    args.use_compound_names = False
    args.features_generator = None
    args.folds_file = None
    args.val_fold_index = None
    args.test_fold_index = None
    args.coord = False
    args.atom_messages = False
    args.features_only = False
    args.dense = False
    args.self_attention = False
    # ???
    args.num_lrs = 1

    args.fine_tune_coff = 1
    args.no_cache = True
    args.cuda = True

    from rdkit import RDLogger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    log_dir = os.path.join(args.save_dir, "logs")
    logger = create_logger(name='train', save_dir=log_dir, quiet=args.quiet)

    run_training(args=args, logger=logger)


if __name__ == '__main__':
    train()
