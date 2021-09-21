import os
import time
import pandas as pd
from service.grpc_util import grpc_server, grpc_client
from argparse import ArgumentParser


def get_available_tasks():
    return ['esol', 'bbbp', 'tox21', 'caco2', 'pka', 'hia', 'ppb', 'pgp-substrate',
            'pgp-inhibitor', 'herg', 'logd', 'cyp', 'hlm', 'rlm', 'mlm', 'ht21',
            'rt21', 'mt21']


def add_predict_args(parser: ArgumentParser):
    parser.add_argument('--role', type=str, default='server',
                        help='Role(server, client)')
    parser.add_argument('--ip', type=str, default='9.19.177.76',
                        help='Network ip')
    parser.add_argument('--port', type=str, default='8888',
                        help='Network port')
    parser.add_argument('--max_workers', type=int, default=1,
                        help='Server: max worker num (for server)')
    parser.add_argument('--tasks', nargs='+', default=get_available_tasks(),
                        help='Tasks of molecular property prediction (for client)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use gpu (for server)')
    parser.add_argument('--group_size', type=int, default=2,
                        help='The group size of tasks (for server)')
    parser.add_argument('--input_file', type=str,
                        help='Path of testing csv (for client)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found) (for server)')


def main():
    parser = ArgumentParser(description='Service of Molecular Property Prediction')
    add_predict_args(parser)
    args = parser.parse_args()

    if args.role == 'server':
        server = grpc_server.Server(ip=args.ip,
                                    port=args.port,
                                    max_workers=args.max_workers,
                                    args=args,
                                    tasks=get_available_tasks(),
                                    gpu=args.gpu,
                                    group_size=args.group_size,
                                    checkpoint_dir=args.checkpoint_dir)
        server.start()
    else:
        if os.environ.get('http_proxy'):
            del os.environ['http_proxy']
        if os.environ.get('https_proxy'):
            del os.environ['https_proxy']
        client = grpc_client.Client(ip=args.ip, port=args.port)
        df = pd.read_csv(args.input_file)
        smiles = [str(smile) for smile in df['smiles']]
        tic = time.time()
        res = client.get_mole_prop(tasks=args.tasks, smiles=smiles)
        toc = time.time()
        print('get molecular property ({}): {}\ncost time: {}s'.format(args.tasks, res.msg, (toc - tic)))
        for task in res.score:
            print('{}:'.format(task))
            for i, smile in enumerate(res.score[task].task_score):
                print('({}) smile:{} score:{}'.format(i + 1, smile, res.score[task].task_score[smile].val))


if __name__ == '__main__':
    main()
