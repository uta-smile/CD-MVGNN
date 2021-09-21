import argparse

def add_model_args(parser):
    """Add model-related arguments.

    Args:
    * parser: argument parser

    Returns:
    * parser: argument parser with model-related arguments inserted
    """

    model_arg = parser.add_argument_group('Model')
    
    model_arg.add_argument('--hidden_size', 
                           type=int, default=450, 
                           help='Dimension of the hidden vector')
    
    model_arg.add_argument('--latent_size', 
                           type=int, default=56, 
                           help='Dimension of the latent vector: z_tree and z_mol')
    
    model_arg.add_argument('--depthT', 
                           type=int, default=20, 
                           help='Depth of junction tree')
    
    model_arg.add_argument('--depthG', 
                           type=int, default=3, 
                           help='Depth of molecular graph')

    return parser


def add_train_args(parser):
    """Add training-related arguments.

    Args:
    * parser: argument parser

    Returns:
    * parser: argument parser with training-related arguments inserted
    """

    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--train_prep', 
                           required=True, 
                           help='Where to load preprocessed data')
    
    #train_arg.add_argument('--property', required=True)
    train_arg.add_argument('--vocab', 
                           required=True, 
                           help='Where to load cluster vocabulary')
    
    train_arg.add_argument('--train_save_dir', 
                           required=True,
                           help='Where to save model')
    
    train_arg.add_argument('--load_epoch', 
                           type=int, default=0, 
                           help='Where to load model for given epoch')

    train_arg.add_argument('--batch_size', 
                           type=int, default=32, 
                           help='Minibatch size')

    train_arg.add_argument('--lr', 
                           type=float, default=1e-3,
                           help='Learning rate')

    train_arg.add_argument('--clip_norm', 
                           type=float, default=50.0, 
                           help='Performing gradient clippiing when great than clip_norm')
    
    train_arg.add_argument('--beta', 
                           type=float, default=0.0, 
                           help='Initial value of the weight for the KL term')
    
    train_arg.add_argument('--step_beta', 
                           type=float, default=0.001, 
                           help='Incremental increase added to Beta')
    
    train_arg.add_argument('--max_beta', 
                           type=float, default=1.0, 
                           help='Max allowed value for beta')
    
    train_arg.add_argument('--warmup', 
                           type=int, default=40000,
                           help='Warmming up')

    train_arg.add_argument('--epoch', 
                           type=int, default=20, 
                           help='Number of training epoches')
    
    train_arg.add_argument('--anneal_rate', 
                           type=float, default=0.9,
                           help='Anneal rate')
    
    train_arg.add_argument('--anneal_iter', 
                           type=int, default=40000, 
                           help='Anneal iter')
    
    train_arg.add_argument('--kl_anneal_iter', 
                           type=int, default=1000, 
                           help='Anneal iteration for KL term')
    
    train_arg.add_argument('--print_iter', 
                           type=int, default=50, 
                           help='Number of iter for printing')
    
    train_arg.add_argument('--save_iter', 
                           type=int, default=5000, 
                           help='How many iters to save model once')

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
