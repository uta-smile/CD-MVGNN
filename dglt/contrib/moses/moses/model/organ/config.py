import argparse
from dglt.contrib.moses.moses.model.organ.metrics_reward import MetricsReward


def add_model_args(parser):
    def restricted_float(arg):
        if float(arg) < 0 or float(arg) > 1:
            raise argparse.ArgumentTypeError(
                '{} not in range [0, 1]'.format(arg)
            )
        return float(arg)

    def conv_pair(arg):
        if arg[0] != '(' or arg[-1] != ')':
            raise argparse.ArgumentTypeError('Wrong pair: {}'.format(arg))

        feats, kernel_size = arg[1:-1].split(',')
        feats, kernel_size = int(feats), int(kernel_size)

        return feats, kernel_size

    if parser is None:
        parser = argparse.ArgumentParser()

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--embedding_size', type=int, default=32,
                           help='Embedding size in generator '
                                'and discriminator')
    model_arg.add_argument('--hidden_size', type=int, default=512,
                           help='Size of hidden state for lstm '
                                'layers in generator')
    model_arg.add_argument('--num_layers', type=int, default=2,
                           help='Number of lstm layers in generator')
    model_arg.add_argument('--dropout', type=float, default=0,
                           help='Dropout probability for lstm '
                                'layers in generator')
    model_arg.add_argument('--discriminator_layers', nargs='+', type=conv_pair,
                           default=[(100, 1), (200, 2), (200, 3),
                                    (200, 4), (200, 5), (100, 6),
                                    (100, 7), (100, 8), (100, 9),
                                    (100, 10), (160, 15), (160, 20)],
                           help='Numbers of features for convalution '
                                'layers in discriminator')
    model_arg.add_argument('--discriminator_dropout', type=float, default=0,
                           help='Dropout probability for discriminator')
    model_arg.add_argument('--reward_weight', type=restricted_float,
                           default=0.7,
                           help='Reward weight for policy gradient training')
    return parser


def add_train_args(parser):
    """Add training-related arguments.

    Args:
    * parser: argument parser

    Returns:
    * parser: argument parser with training-related arguments inserted
    """

    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--generator_pretrain_epochs', type=int,
                           default=50,
                           help='Number of epochs for generator pretraining')
    train_arg.add_argument('--discriminator_pretrain_epochs', type=int,
                           default=50,
                           help='Number of epochs for '
                                'discriminator pretraining')
    train_arg.add_argument('--pg_iters', type=int, default=1000,
                           help='Number of inerations for policy '
                                'gradient training')
    train_arg.add_argument('--n_batch', type=int, default=512,
                           help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=1e-4,
                           help='Learning rate')
    train_arg.add_argument('--n_jobs', type=int, default=8,
                           help='Number of threads')

    train_arg.add_argument('--max_length', type=int, default=278,
                           help='Maximum length for sequence')
    train_arg.add_argument('--clip_grad', type=float, default=5,
                           help='Clip PG generator gradients to this value')
    train_arg.add_argument('--rollouts', type=int, default=128,
                           help='Number of rollouts')
    train_arg.add_argument('--generator_updates', type=int, default=1,
                           help='Number of updates of generator per iteration')
    train_arg.add_argument('--discriminator_updates', type=int, default=1,
                           help='Number of updates of discriminator '
                                'per iteration')
    train_arg.add_argument('--discriminator_epochs', type=int, default=10,
                           help='Number of epochs of discriminator '
                                'per iteration')
    train_arg.add_argument('--pg_smooth_const', type=float, default=0.1,
                           help='Smoothing factor for Policy Gradient logs')

    parser.add_argument('--n_ref_subsample', type=int, default=500,
                        help='Number of reference molecules '
                             '(sampling from training data)')
    parser.add_argument('--additional_rewards', nargs='+', type=str,
                        choices=MetricsReward.supported_metrics, default=[],
                        help='Adding of addition rewards')
    return parser

def add_generate_args(parser):
    """Add generation-related arguments.

    Args:
    * parser: argument parser

    Returns:
    * parser: argument parser with generation-related arguments inserted
    """

    generate_arg = parser.add_argument_group('Generate')
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

