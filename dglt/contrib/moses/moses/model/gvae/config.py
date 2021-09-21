import argparse

def add_model_args(parser):
    """Add model-related arguments.

    Args:
    * parser: argument parser

    Returns:
    * parser: argument parser with model-related arguments inserted
    """

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--q_cell',
                           type=str, default='gru', choices=['gru'],
                           help='Encoder rnn cell type')
    model_arg.add_argument('--q_bidir',
                           default=False, action='store_true',
                           help='If to add second direction to encoder')
    model_arg.add_argument('--q_d_h',
                           type=int, default=256,
                           help='Encoder h dimensionality')
    model_arg.add_argument('--q_n_layers',
                           type=int, default=1,
                           help='Encoder number of layers')
    model_arg.add_argument('--q_dropout',
                           type=float, default=0.5,
                           help='Encoder layers dropout')
    model_arg.add_argument('--d_cell',
                           type=str, default='gru', choices=['gru'],
                           help='Decoder rnn cell type')
    model_arg.add_argument('--d_n_layers',
                           type=int, default=3,
                           help='Decoder number of layers')
    model_arg.add_argument('--d_dropout',
                           type=float, default=0,
                           help='Decoder layers dropout')
    model_arg.add_argument('--d_z',
                           type=int, default=56,
                           help='Latent vector dimensionality')
    model_arg.add_argument('--d_d_h',
                           type=int, default=501,
                           help='Latent vector dimensionality')
    model_arg.add_argument('--d_emb',
                           type=int, default=100,
                           help='Word embedding vector dimensionality')
    model_arg.add_argument('--freeze_embeddings',
                           default=False, action='store_true',
                           help='If to freeze embeddings while training')
    model_arg.add_argument('--smiles_maxlen',
                           type=int, default=278,
                           help='Maximal length of a SMILES string.')

    return parser


def add_train_args(parser):
    """Add training-related arguments.

    Args:
    * parser: argument parser

    Returns:
    * parser: argument parser with training-related arguments inserted
    """

    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--n_batch',
                           type=int, default=512,
                           help='Batch size')
    train_arg.add_argument('--r_tval_samples',
                           type=float, default=0.1,
                           help='Ratio of trn-val samples')
    train_arg.add_argument('--n_epoch',
                           type=int, default=50,
                           help='Number of epochs')
    train_arg.add_argument('--lr_init',
                           type=float, default=1e-3,
                           help='Initial learning rate')
    train_arg.add_argument('--lr_min',
                           type=float, default=1e-5,
                           help='Minimal learning rate')
    train_arg.add_argument('--lr_factor',
                           type=float, default=0.2,
                           help='Reducing factor of learning rate')
    train_arg.add_argument('--n_last',
                           type=int, default=1000,
                           help='Number of iters to smooth loss calc')
    train_arg.add_argument('--rule_path',
                           type=str, default='./moses/model/gvae/grules.txt',
                           help='Path to the Char/Grammar rule file')
    train_arg.add_argument('--clip_grad',
                           type=int, default=50,
                           help='Clip gradients to this value')
    train_arg.add_argument('--kl_start',
                           type=int, default=0,
                           help='Epoch to start change kl weight from')
    train_arg.add_argument('--kl_w_start',
                           type=float, default=0,
                           help='Initial kl weight value')
    train_arg.add_argument('--kl_w_end',
                           type=float, default=1,
                           help='Maximum kl weight value')
    train_arg.add_argument('--lr_start',
                           type=float, default=3 * 1e-4,
                           help='Initial lr value')
    train_arg.add_argument('--lr_n_period',
                           type=int, default=10,
                           help='Epochs before first restart in SGDR')
    train_arg.add_argument('--lr_n_restarts',
                           type=int, default=6,
                           help='Number of restarts in SGDR')
    train_arg.add_argument('--lr_n_mult',
                           type=int, default=1,
                           help='Mult coefficient after restart in SGDR')
    train_arg.add_argument('--lr_end',
                           type=float, default=3 * 1e-4,
                           help='Maximum lr weight value')
    train_arg.add_argument('--n_jobs',
                           type=int, default=1,
                           help='Number of threads')
    train_arg.add_argument('--n_workers',
                           type=int, default=1,
                           help='Number of workers for DataLoaders')

    return parser


def add_generate_args(parser):
    """Add generation-related arguments.

    Args:
    * parser: argument parser

    Returns:
    * parser: argument parser with generation-related arguments inserted
    """

    generate_arg = parser.add_argument_group('Generate')

    return parser


def get_train_parser(parser=None):
    """Get an argument parser for training.

    Args:
    * parser: argument parser

    Returns:
    * parser: argument parser with model/training-related arguments inserted
    """

    if parser is None:
        parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_train_args(parser)

    return parser


def get_generate_parser(parser=None):
    """Get an argument parser for generation.

    Args:
    * parser: argument parser

    Returns:
    * parser: argument parser with model/generation-related arguments inserted
    """

    if parser is None:
        parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_generate_args(parser)

    return parser
