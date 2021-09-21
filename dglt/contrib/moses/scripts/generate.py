import argparse
import os
import sys
import math
import torch
import rdkit

import pandas as pd
import warnings
from tqdm import tqdm
from collections import Counter
from itertools import repeat
from dglt.contrib.moses.moses.utils import valid_smiles
from dglt.contrib.moses.moses.metrics import mol_passes_filters

from dglt.contrib.moses.moses.script_utils import add_generate_args, read_smiles_csv, \
    read_props_dict, set_seed, preprocess_config
from dglt.contrib.moses.moses.models_storage import ModelsStorage

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()


def get_parser():
    """Get an argument parser for SMILES generation.

    Returns:
    * parser: argument parser
    """

    parser = argparse.ArgumentParser(conflict_handler='resolve')
    subparsers = parser.add_subparsers(
        title='Models generator script', description='available models')
    for model_name in MODELS.get_model_names():
        model_generate_parser = MODELS.get_model_generate_parser(model_name)
        if model_generate_parser is not None:
            model_generate_parser(add_generate_args(
                subparsers.add_parser(model_name, conflict_handler='resolve')))
        else:
            print('[WARNING] argument parser not set for SMILES generation with ' + model_name)

    return parser


def sample(generator, config):
    """Sample a list of SMILES sequences.

    Args:
    * generator: SMILES generator
    * config: configurations

    Returns:
    * smiles_list: list of SMILES sequences
    """

    assert config.n_samples is not None, \
        '<--n_samples> must be specified when <--gen_mode> is \'sample\''

    if config.unique:
        smiles_list = set()
        list_len = 0
    else:
        smiles_list = []

    tot_batch = 0
    with tqdm(total=config.n_samples, desc='Sampling SMILES sequences') as T:
        while tot_batch < config.n_samples:
            # sample a mini-batch of SMILES sequences
            if config.n_samples - tot_batch > config.n_batch:
                smiles_list_new = generator.sample(config.n_batch, config.max_len)
            else:
                n_samples_left = config.n_samples - tot_batch
                smiles_list_new = generator.sample(n_samples_left, config.max_len)

            # append to the list
            if config.valid_samples:
                smiles_list_new = [valid_smiles(_) for _ in smiles_list_new
                                   if valid_smiles(_) is not None]
            if config.unique:
                smiles_list.update(smiles_list_new)
                list_len = len(smiles_list) - list_len
                T.update(list_len)
                tot_batch += list_len
                list_len = len(smiles_list)
            else:
                smiles_list.extend(smiles_list_new)
                T.update(len(smiles_list_new))
                tot_batch += len(smiles_list_new)

    if config.unique:
        return list(smiles_list)
    return smiles_list


def recon(generator, config):
    """Reconstruct a list of SMILES sequences.

    Args:
    * generator: SMILES generator
    * config: configurations

    Returns:
    * smiles_list: list of SMILES sequences
    """

    assert config.recon_load is not None, \
        '<--recon_load> must be specified when <--gen_mode> is \'recon\''

    assert len(config.csv_col_names) < 3, \
        '<--csv_col_names> must contain less than 3 columns when <--gen_mode> is \'recon\''
    config.n_samples = config.nsamples if config.n_samples is not None else 1

    # read in a list of SMILES sequences to be reconstructed
    smiles_list_in = read_smiles_csv(config.recon_load, config.csv_col_names)

    # reconstruct candidates of SMILES sequences
    smiles_list_out = generator.recon(smiles_list_in, config.max_len)

    # Filter the candidates
    smiles_data = repeat(None)
    if config.remove_original_smiles:
        if not isinstance(smiles_list_in, (list, tuple)) and \
                        'SMILES' not in smiles_list_in.columns:
            warnings.warn("Can not find SMILES column in the data. "
                          "Please check <--cse_col_names> includes 'SMILES' column.")
            config.remove_original_smiles = False
        else:
            if isinstance(smiles_list_in, (list, tuple)):
                smiles_data = smiles_list_in
            else:
                smiles_data = smiles_list_in['SMILES'].values

    # Remove original smiles
    if config.remove_original_smiles:
        smiles_list = []
        for smiles, candidates in zip(smiles_data, smiles_list_out):
            smiles_list.append(list(filter(smiles.__ne__, candidates)))
    else:
        smiles_list = smiles_list_out

    # Get most frequent candidates
    if config.keep_all_candidates:
        smiles_results = []
        for candidates in smiles_list:
            if config.valid_samples:
                res = [valid_smiles(smiles_ele) for smiles_ele in candidates
                       if valid_smiles(smiles_ele) is not None]
            else:
                res = candidates
            smiles_results.append(res)
        smiles_titles = ["SMILES_%d" % _ for _ in range(max(map(len, smiles_results)))]
    else:
        smiles_results = []
        for candidates in smiles_list:
            cnt = Counter()
            for smiles_ele in candidates:
                if valid_smiles(smiles_ele) is not None:
                    cnt[valid_smiles(smiles_ele)] += 1
            smiles_results.append(cnt.most_common(1)[0][0])
        smiles_titles = ['SMILES']

    return smiles_results, smiles_titles


def design(generator, config):
    """Design a list of SMILES sequences that satisfy given property values.

    Args:
    * generator: SMILES generator
    * config: configurations

    Returns:
    * smiles_list: list of SMILES sequences
    """
    assert config.design_load is not None or len(config.design_json) > 0, \
    'Either <--design_load> or <--design_json> must be specified when <--gen_mode> is \'design\''
    assert len(generator.model.design_col_names) > 0, \
    'Loaded model is not trained for design mode. Please use <--gen_mode sample>, instead.'

    all_cols = True
    if len(config.design_json) > 0:
        config.design_col_names = list(config.design_json.keys())
        all_cols = False
    model_cols = generator.model.design_col_names
    config = preprocess_config(config, config.design_load,
                               all_cols=all_cols, model_cols=model_cols)

    if len(config.design_json) > 0:
        design_list = read_props_dict(config.design_json, config.design_col_names)
    else:
        design_list = read_smiles_csv(config.design_load, config.design_col_names)


    design_list = pd.concat([pd.DataFrame([], columns=model_cols), design_list], sort=False).values

    # generate a list of SMILES sequences via designed properties
    return generator.design(config.n_samples, design_list, config.max_len)

    # assert config.n_samples is not None, \
    #     '<--n_samples> must be specified when <--gen_mode> is \'design\''
    #
    # properties = {
    #     'logP': 2.5,
    #     'SA': None,
    #     'NP': None,
    #     'QED': None,
    #     'weight': None,
    # }
    #
    # smiles_list = []
    # n_batches = int(math.ceil(config.n_samples / config.n_batch))
    # with tqdm(total=config.n_samples, desc='Designing SMILES sequences') as T:
    #     for idx_batch in range(n_batches):
    #         # design a mini-batch of SMILES sequences
    #         if idx_batch < n_batches - 1:
    #             smiles_list_new = generator.design(config.n_batch, properties, config.max_len)
    #         else:
    #             n_samples_left = config.n_samples - config.n_batch * (n_batches - 1)
    #             smiles_list_new = generator.design(n_samples_left, properties, config.max_len)
    #
    #         # append to the list
    #         smiles_list.extend(smiles_list_new)
    #         T.update(len(smiles_list_new))
    #
    # return smiles_list

def main(model_name, config):
    """Main Entry.

    Args:
    * model_name: model's name
    * config: configurations
    """

    # initialization
    set_seed(config.seed)
    device = torch.device(config.device)
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)

    # restore a pre-trained model and build a generator from it
    model_vocab = torch.load(config.vocab_load)
    model_state = torch.load(config.model_load)
    model_config = torch.load(config.config_load)
    model = MODELS.get_model_class(model_name)(model_vocab, model_config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()  # set the model in the evaluation mode
    generator = MODELS.get_model_generator(model_name)(model, model_config, config)

    # execute the specified generation routine
    if config.gen_mode == 'sample' and \
            (config.design_load is not None or len(config.design_json) > 0):
        smiles_list = design(generator, config)
        smiles_titles = ['SMILES']
    elif config.gen_mode == 'sample':
        smiles_list = sample(generator, config)
        smiles_titles = ['SMILES']
    elif config.gen_mode == 'recon' and \
            (config.design_load is not None or len(config.design_json) > 0):
        raise ValueError('unrecognized generation mode: recon with design')
        # smiles_list = recon_design(generator, config)
    elif config.gen_mode == 'recon':
        smiles_list, smiles_titles = recon(generator, config)
    else:
        raise ValueError('unrecognized generation mode: ' + config.gen_mode)

    # write SMILES sequences to a *.csv file
    samples = pd.DataFrame(smiles_list, columns=smiles_titles)
    samples.to_csv(config.gen_save, index=False)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    model_name = sys.argv[1]
    main(model_name, config)
