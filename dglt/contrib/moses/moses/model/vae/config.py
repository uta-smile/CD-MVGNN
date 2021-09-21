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
                           type=int, default=128,
                           help='Latent vector dimensionality')
    model_arg.add_argument('--d_d_h',
                           type=int, default=512,
                           help='Latent vector dimensionality')
    model_arg.add_argument('--freeze_embeddings',
                           default=False, action='store_true',
                           help='If to freeze embeddings while training')
    model_arg.add_argument("--no_self_loop",
                           dest="self_loop", action="store_false")

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
    train_arg.add_argument('--mse_weight',
                           type=float, default=1,
                           help='MSE weight value')
    train_arg.add_argument('--lr_start',
                           type=float, default=1e-3,
                           help='Initial lr value')
    train_arg.add_argument('--n_epoch',
                           type=int, default=60,
                           help='Epochs')
    train_arg.add_argument('--lr_end',
                           type=float, default=1e-6,
                           help='Maximum lr weight value')
    train_arg.add_argument('--lr_factor',
                           type=float, default=0.5,
                           help='Reduce factor of lr')
    train_arg.add_argument('--lr_patience',
                           type=int, default=3,
                           help='Patience for lr weight')
    train_arg.add_argument('--n_last',
                           type=int, default=1000,
                           help='Number of iters to smooth loss calc')
    train_arg.add_argument('--n_jobs',
                           type=int, default=1,
                           help='Number of threads')
    train_arg.add_argument('--no_norm', dest="auto_norm",
                           action="store_false", default=True,
                           help='Wether to norm the input labels')

    return parser


def add_generate_args(parser):
    """Add generation-related arguments.

    Args:
    * parser: argument parser

    Returns:
    * parser: argument parser with generation-related arguments inserted
    """

    generate_arg = parser.add_argument_group('Generate')
    generate_arg.add_argument('--no_norm', dest="auto_norm",
                              action="store_false", default=True,
                              help='Wether to norm the input labels')
    generate_arg.add_argument('--encode_times',
                              type=int, default=10, metavar='N',
                              help='Encode smile N times.')
    generate_arg.add_argument('--decode_times',
                              type=int, default=5, metavar='M',
                              help='Decode each latent vector M times. There will be N*M candidates.')

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
