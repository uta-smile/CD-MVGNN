import os
from argparse import ArgumentParser
import numpy as np
import math
from rdkit import RDLogger

from dglt.train import make_predictions, write_prediction
from dglt.utils import makedirs
from dglt.parsing import get_newest_train_args
from third_party.dimorphite_dl.acid_base import mol_cls
from Crypto.Cipher import AES

def get_prediction_args():
    parser = ArgumentParser()
    add_predict_args(parser)
    predict_args = parser.parse_args()
    return predict_args

def add_predict_args(parser: ArgumentParser):
    """
    Adds predict arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--task', type=str, help='Which task to carry out', choices=['esol', 'bbbp', 'hia', 'pka', 'caco2', 'tox21', 'ppb', 'pgp-substrate', 'pgp-inhibitor', 'herg'], required=True)
    parser.add_argument('--test_path', type=str,
                        help='Path of testing csv', required=True)
    parser.add_argument('--preds_path', type=str,
                        help='Path of predicts result', required=True)

def update_checkpoint_args(args):
    """
    Walks the checkpoint directory to find all checkpoints, updating args.checkpoint_paths and args.ensemble_size.

    :param args: Arguments.
    """
    if hasattr(args, 'checkpoint_paths') and args.checkpoint_paths is not None:
        return

    if args.checkpoint_dir is not None and args.checkpoint_path is not None:
        raise ValueError('Only one of checkpoint_dir and checkpoint_path can be specified.')

    if args.checkpoint_dir is None:
        args.checkpoint_paths = [args.checkpoint_path] if args.checkpoint_path is not None else None
        return

    args.checkpoint_paths = []

    for root, _, files in os.walk(args.checkpoint_dir):
        for fname in files:
            if fname.endswith('.pt'):
                args.checkpoint_paths.append(os.path.join(root, fname))

    args.ensemble_size = len(args.checkpoint_paths)

    if args.ensemble_size == 0:
        raise ValueError('Failed')

def modify_predict_args(args):
    """
    Modifies and validates predicting args in place.

    :param args: Arguments.
    """
    assert args.test_path
    assert args.preds_path

    # single model
    model_path = './model_enc/model_' + args.task
    if os.path.exists(model_path):
        args.checkpoint_dir = model_path
        args.checkpoint_path = None

    update_checkpoint_args(args)
    assert args.checkpoint_paths
    args.cuda = False
    args.deploy = True

    # Create directory for preds path
    makedirs(args.preds_path, isfile=True)


def run_prediction(sub_task=None):
    predict_args = get_prediction_args()
    if sub_task is not None:
        predict_args.task = sub_task
    modify_predict_args(predict_args)
    newest_train_args = get_newest_train_args()
    avg_preds, test_smiles = make_predictions(predict_args, newest_train_args)
    return avg_preds, test_smiles, predict_args


if __name__ == '__main__':
    lg = RDLogger.logger()
    RDLogger.DisableLog('rdApp.*')
    lg.setLevel(RDLogger.CRITICAL)

    predict_args = get_prediction_args()

    if predict_args.task == 'pka':
        avg_preds, test_smiles, modified_args = run_prediction()
        acid_base = mol_cls(test_smiles)
        avg_preds[acid_base == None] = np.nan
        modified_args.task_names += ['acidic_site', 'basic_site']
        avg_preds = np.column_stack((avg_preds, acid_base))
        write_prediction(avg_preds, test_smiles, modified_args)
    elif predict_args.task == 'caco2':
        avg_preds, test_smiles, modified_args = run_prediction()
        # avg_preds = (np.power(10, avg_preds.tolist()) - 1) / 10
        for i in range(avg_preds.shape[0]):
            if avg_preds[i] is not None:
                for j in range(len(avg_preds[i])):
                    avg_preds[i][j] = (math.pow(10, avg_preds[i][j]) - 1) / 10
        write_prediction(avg_preds, test_smiles, modified_args)
    elif predict_args.task == 'ppb':
        avg_preds, test_smiles, modified_args = run_prediction()
        avg_preds[avg_preds > 1] = 1
        avg_preds[avg_preds < 0] = 0
        write_prediction(avg_preds, test_smiles, modified_args)
    else:
        avg_preds, test_smiles, modified_args = run_prediction()
        write_prediction(avg_preds, test_smiles, modified_args)