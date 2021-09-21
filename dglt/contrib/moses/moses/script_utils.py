import argparse
import random
import re
import numpy as np
import pandas as pd
import torch
import json
import warnings
from dglt.multi_gpu_wrapper import MultiGpuWrapper as mgw


class JsonAction(argparse.Action):
    # Check if the input json is valid and convert it to dict
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, json.loads(' '.join(values).replace("'", "\"")))
        except json.decoder.JSONDecodeError:
            raise argparse.ArgumentError(
                self,
                "invalid json value: %s" % ' '.join(values).replace("'", "\"")
            )

def add_common_arg(parser):
    def torch_device(arg):
        if re.match('^(cuda(:[0-9]+)?|cpu)$', arg) is None:
            raise argparse.ArgumentTypeError(
                'Wrong device format: {}'.format(arg)
            )

        if arg != 'cpu':
            splited_device = arg.split(':')

            if (not torch.cuda.is_available()) or \
                    (len(splited_device) > 1 and
                     int(splited_device[1]) > torch.cuda.device_count()):
                raise argparse.ArgumentTypeError(
                    'Wrong device: {} is not available'.format(arg)
                )

        return arg

    # Base
    parser.add_argument('--device',
                        type=torch_device, default='cuda',
                        help='Device to run: "cpu" or "cuda:<device number>"')
    parser.add_argument('--seed',
                        type=int, default=0,
                        help='Seed')
    parser.add_argument('--n_workers',
                        type=int, default=1,
                        help='Number of workers for DataLoaders')

    return parser


def add_train_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--train_load',
                            type=str, required=True,
                            help='Input data in csv format to train')
    common_arg.add_argument('--val_load', type=str,
                            help="Input data in csv format to validation")
    common_arg.add_argument('--model_save',
                            type=str, required=True, default='model.pt',
                            help='Where to save the model')
    common_arg.add_argument('--model_load',
                            type=str, default=None,
                            help='Where to load the warm-start model')
    common_arg.add_argument('--fine_tune',
                            type=str, choices=['all', 'fix_enc', 'fix_emb'],
                            default=['all'], nargs='+',
                            help='Use fine tune model. <--model_load> should not be None')
    common_arg.add_argument('--save_frequency',
                            type=int, default=20,
                            help='How often to save the model')
    common_arg.add_argument('--log_file',
                            type=str, required=False,
                            help='Where to save the log')
    common_arg.add_argument('--config_save',
                            type=str, required=True,
                            help='Where to save the config')
    common_arg.add_argument('--vocab_save',
                            type=str,
                            help='Where to save the vocab')
    common_arg.add_argument('--vocab_load',
                            type=str,
                            help='Where to load the vocab; '
                                 'otherwise it will be evaluated')
    common_arg.add_argument('--extra_vocab',
                            type=str, nargs='+', metavar='V', default=None,
                            help='Add extra vocab here')
    common_arg.add_argument('--csv_col_names',
                            type=str, default=['SMILES'], nargs='+',
                            help='Column names to be loaded from *.csv file')
    common_arg.add_argument('--enable_multi_gpu',
                            action='store_true',
                            help='Enable multi gpu process.')
    design_group_arg = common_arg.add_mutually_exclusive_group()
    design_group_arg.add_argument('--design_col_names',
                                  type=str, default=[], nargs='+',
                                  help="Column names to be used as design labels")
    design_group_arg.add_argument('--design_all_cols',
                                  action="store_true", default=False,
                                  help="All columns except <--csv_col_names> to be used as design labels")

    return parser


def add_sample_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--model_load',
                            type=str, required=True,
                            help='Where to load the model')
    common_arg.add_argument('--config_load',
                            type=str, required=True,
                            help='Where to load the config')
    common_arg.add_argument('--vocab_load',
                            type=str, required=True,
                            help='Where to load the vocab')
    common_arg.add_argument('--n_samples',
                            type=int, required=True,
                            help='Number of samples to sample')
    common_arg.add_argument('--gen_save',
                            type=str, required=True,
                            help='Where to save the gen molecules')
    common_arg.add_argument("--n_batch",
                            type=int, default=32,
                            help="Size of batch")
    common_arg.add_argument("--max_len",
                            type=int, default=278,
                            help="Max of length of SMILES")


    return parser

def add_recon_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--model_load',
                            type=str, required=True,
                            help='Where to load the model')
    common_arg.add_argument('--config_load',
                            type=str, required=True,
                            help='Where to load the config')
    common_arg.add_argument('--vocab_load',
                            type=str, required=True,
                            help='Where to load the vocab')
    common_arg.add_argument('--recon_load',
                            type=str, required=True,
                            help='Input data in csv format to reconstruct')
    common_arg.add_argument('--csv_col_name',
                            type=str, default='SMILES',
                            help='Name of the column to be read from *.csv file')
    common_arg.add_argument('--gen_save',
                            type=str, required=True,
                            help='Where to save the gen molecules')
    common_arg.add_argument("--n_batch",
                            type=int, default=32,
                            help="Size of batch")
    common_arg.add_argument("--max_len",
                            type=int, default=278,
                            help="Max of length of SMILES")

    return parser


def add_generate_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--model_load',
                            type=str, required=True,
                            help='Where to load the model')
    common_arg.add_argument('--config_load',
                            type=str, required=True,
                            help='Where to load the config')
    common_arg.add_argument('--vocab_load',
                            type=str, required=True,
                            help='Where to load the vocab')
    common_arg.add_argument('--gen_save',
                            type=str, required=True,
                            help='Where to save the gen molecules')
    common_arg.add_argument('--gen_mode',
                            type=str, required=True,
                            choices=['sample', 'recon'],
                            help='Generation mode: \'sample\' / \'recon\'')
    common_arg.add_argument('--n_samples',
                            type=int, default=None,
                            help='Number of samples to sample')
    common_arg.add_argument('--recon_load',
                            type=str, default=None,
                            help='Where to load SMILES sequences to be reconstructed')
    common_arg.add_argument("--n_batch",
                            type=int, default=32,
                            help="Size of batch")
    common_arg.add_argument("--max_len",
                            type=int, default=278,
                            help="Max of length of SMILES")
    common_arg.add_argument('--csv_col_names',
                            type=str, default=['SMILES'], nargs='+',
                            help='Column names to be loaded from *.csv file')
    common_arg.add_argument('--remove_original_smiles',
                            action="store_true",
                            help="Remove the original smiles form reconstructing candidates")
    common_arg.add_argument('--keep_all_candidates',
                            action="store_true",
                            help="Keep all reconstructing candidates instead of pick up the most frequent one")
    common_arg.add_argument('--ignore_valid',
                            action="store_false", dest="valid_samples",
                            help="Check if samples are valid")
    common_arg.add_argument('--unique',
                            action='store_true',
                            help="Only keep unique SMILES")
    design_group_arg = common_arg.add_mutually_exclusive_group()
    design_group_arg.add_argument('--design_load',
                                  type=str, default=None,
                                  help="Where to load designed properties to be generated.")
    design_group_arg.add_argument('--design_json',
                                  action=JsonAction, default={}, nargs="+",
                                  help="Designed properties to be generated.")
    common_arg.add_argument('--design_col_names',
                            type=str, default=[], nargs='+',
                            help="Column names to be used as design labels")

    return parser


def preprocess_config(config, path, all_cols=True, model_cols=None, modify_config=None):
    """Preprocess the config

    Args:
    * config: original config
    * path: path to load column names

    Returns:
    * config: preprocess config.
              If design_all_cols is true, then read col_names from train_load
              If model_cols is not None, pick all design_col_names that belongs to model_cols
              Add design_col_names to the csv_col_names in order and remove duplicates
    """
    if config.csv_col_names is None:
        config.csv_col_names = []

    if all_cols:
        config.design_col_names = list(pd.read_csv(path, nrows=1).columns)
        for col in config.csv_col_names:
            if col in config.design_col_names:
                config.design_col_names.remove(col)

    if model_cols is not None:
        tmp_cols = model_cols.copy()
        for col in model_cols:
            if col not in config.design_col_names:
                tmp_cols.remove(col)
        ext_set = set(config.design_col_names) - set(tmp_cols)
        if len(ext_set) > 0:
            warnings.warn("[%s] are not in the model properties. "
                          "Ignored." % ','.join(ext_set))

        config.design_col_names = tmp_cols

    if config.design_col_names is not None:
        for col in config.design_col_names:
            if col in config.csv_col_names:
                config.csv_col_names.remove(col)
        config.csv_col_names += config.design_col_names.copy()

    config.master_worker = (mgw.rank() == 0) if config.enable_multi_gpu else True
    if modify_config is not None:
        modify_config(config)

    return config

def read_smiles_csv(path, col_names=None):
    """Read SMILES and/or other columns from *.csv file.

    Args:
    * path: path to the *.csv file
    * col_names: a list of column names

    Returns:
    * csv_data: list (for single-column) OR pandas.core.frame.DataFrame (for multi-column)
    """
    if col_names is None:
        col_names = ['SMILES']

    # if len(col_names) == 1:
    #     csv_data = pd.read_csv(path, usecols=col_names, squeeze=True).astype(str).tolist()
    # else:
    #     csv_data = pd.read_csv(path, usecols=col_names)[col_names]
    csv_data = pd.read_csv(path, usecols=col_names)[col_names]

    return csv_data

def read_props_dict(dict, col_names):
    """Read properties from a dictionary

    Args:
    * dict: a dictionary contains property names and designed values
    * col_names: a list of column names

    Returns:
    * dict_data: list (for single-column) OR pandas.core.frame.DataFrame (for multi-column)
    """
    for k, v in dict.items():
        if not isinstance(v, (tuple, list)):
            dict[k] = [v]

    return pd.DataFrame.from_dict(dict)[col_names]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
