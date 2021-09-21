import argparse
import os
import sys
import torch
import rdkit

from dglt.multi_gpu_wrapper import MultiGpuWrapper as mgw
from dglt.contrib.moses.moses.script_utils import add_train_args, read_smiles_csv, set_seed, preprocess_config
from dglt.contrib.moses.moses.models_storage import ModelsStorage

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()


def get_parser():
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    subparsers = parser.add_subparsers(
        title='Models trainer script', description='available models'
    )
    for model in MODELS.get_model_names():
        MODELS.get_model_train_parser(model)(add_train_args(
            subparsers.add_parser(model, conflict_handler='resolve')))
    return parser


def main(model, config):
    # process config
    config = preprocess_config(config, config.train_load,
                               all_cols=config.design_all_cols,
                               modify_config=MODELS.get_modify_config(model))
    if config.config_save is not None:
        torch.save(config, config.config_save)

    # set random seed
    set_seed(config.seed)

    # enable multi gpus
    device = torch.device(config.device)
    if config.enable_multi_gpu:
        mgw.init()
        idx = mgw.local_rank()
        torch.cuda.set_device(idx)
        device = torch.device('cuda:%d' % idx)
    # For CUDNN to work properly
    elif device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)

    train_data = read_smiles_csv(config.train_load, config.csv_col_names)
    if config.val_load:
        val_data = read_smiles_csv(config.val_load, config.csv_col_names)
    else:
        val_data = None
    trainer = MODELS.get_model_trainer(model)(config)

    if config.vocab_load is not None:
        assert os.path.exists(config.vocab_load), \
            'vocab_load path does not exist!'
        vocab = torch.load(config.vocab_load)
    else:
        vocab = trainer.get_vocabulary(train_data, config.extra_vocab)

    if config.vocab_save is not None:
        torch.save(vocab, config.vocab_save)

    model = MODELS.get_model_class(model)(vocab, config)
    if config.model_load is not None:
        model_state = torch.load(config.model_load)
        model.load_state_dict(model_state)
    model = model.to(device)
    trainer.fit(model, train_data, val_data)

    model = model.to('cpu')
    torch.save(model.state_dict(), config.model_save)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    model = sys.argv[1]
    main(model, config)
