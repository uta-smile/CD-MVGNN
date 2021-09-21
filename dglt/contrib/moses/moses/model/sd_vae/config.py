import argparse

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    model_opt = parser.add_argument_group('Model')
    model_opt.add_argument('--grammar_file', default="data/grammar/mol_zinc.grammar", help='grammar production rules')
    model_opt.add_argument('--info_folder', default=None, help='folder of grammar production rules')
    model_opt.add_argument('--encoder_type', default='cnn', help='choose encoder from [tree_lstm | s2v | cnn]')
    model_opt.add_argument('--ae_type', default='vae', help='choose ae arch from [autoenc | vae]')
    model_opt.add_argument('--rnn_type', default='gru', help='choose rnn cell from [gru | sru]')
    model_opt.add_argument('--q_bidir', default=False, action='store_true', help='If to add second direction to encoder')
    model_opt.add_argument('--q_d_h', type=int, default=256, help='Encoder h dimensionality')
    model_opt.add_argument('--q_n_layers', type=int, default=1, help='Encoder number of layers')
    model_opt.add_argument('--q_dropout', type=float, default=0.5, help='Encoder layers dropout')
    model_opt.add_argument('--d_n_layers', type=int, default=3, help='Decoder number of layers')
    model_opt.add_argument('--d_dropout', type=float, default=0, help='Decoder layers dropout')
    model_opt.add_argument('--d_z', type=int, default=128, help='Latent vector dimensionality')
    model_opt.add_argument('--d_d_h', type=int, default=512, help='Latent vector dimensionality')
    model_opt.add_argument('--loss_type', default='vanilla', help='choose loss from [perplexity | binary | vanilla]')
    model_opt.add_argument('--max_decode_steps', type=int, default=278, help='maximum steps for making decoding decisions')
    model_opt.add_argument('--skip_deter', type=int, default=0, help='skip deterministic position')
    model_opt.add_argument('--bondcompact', type=int, default=0, help='compact ringbond representation or not')

    train_opt = parser.add_argument_group('Train')
    train_opt.add_argument('--freeze_embeddings',
                           default=False, action='store_true',
                           help='If to freeze embeddings while training')
    train_opt.add_argument('--n_batch',
                           type=int, default=512,
                           help='minibatch size')
    train_opt.add_argument('--clip_grad',
                           type=int, default=50,
                           help='Clip gradients to this value')
    train_opt.add_argument('--kl_start',
                           type=int, default=0,
                           help='Epoch to start change kl weight from')
    train_opt.add_argument('--kl_w_start',
                           type=float, default=0,
                           help='Initial kl weight value')
    train_opt.add_argument('--kl_w_end',
                           type=float, default=1,
                           help='Maximum kl weight value')
    train_opt.add_argument('--mse_weight',
                           type=float, default=1,
                           help='MSE weight value')
    train_opt.add_argument('--lr_start',
                           type=float, default=1e-3,
                           help='Initial lr value')
    train_opt.add_argument('--lr_n_period',
                           type=int, default=10,
                           help='Epochs before first restart in SGDR')
    train_opt.add_argument('--lr_n_restarts',
                           type=int, default=6,
                           help='Number of restarts in SGDR')
    train_opt.add_argument('--lr_n_mult',
                           type=int, default=1,
                           help='Mult coefficient after restart in SGDR')
    train_opt.add_argument('--lr_end',
                           type=float, default=1e-6,
                           help='Maximum lr weight value')
    train_opt.add_argument('--lr_factor',
                           type=float, default=0.5,
                           help='Reduce factor of lr')
    train_opt.add_argument('--lr_patience',
                           type=int, default=3,
                           help='Patience for lr weight')
    train_opt.add_argument('--prob_fix',
                           type=float, default=0,
                           help='numerical problem')
    train_opt.add_argument('--eps_std',
                           type=float, default=0.01,
                           help='the standard deviation used in reparameterization tric')
    train_opt.add_argument('--n_last',
                           type=int, default=1000,
                           help='Number of iters to smooth loss calc')

    generate_opt = parser.add_argument_group('Generate')
    generate_opt.add_argument('--no_random', action='store_false', dest='use_random', help='Do not use random to generate new smiles.')
    generate_opt.add_argument('--encode_times', type=int, default=10, metavar='N', help='Encode smile N times.')
    generate_opt.add_argument('--decode_times', type=int, default=5, metavar='M', help='Decode each latent vector M times. There will be N*M candidates.')

    preprocess_opt = parser.add_argument_group('Preprocess')
    preprocess_opt.add_argument('--data_dump', help='location of h5 file')
    preprocess_opt.add_argument('--save_as_csv',action="store_true", default=False, help='save preprocesed data as csv file')

    return parser


# cmd_args, _ = get_parser().parse_known_args()
