"""Trains a model on a dataset."""

import os
import random
import json
import numpy as np
import torch
from rdkit import RDLogger

from dglt.parsing import parse_train_args
from dglt.train.prediction import cross_validate
from dglt.train.prediction.utils import job_completed
from dglt.utils import create_logger


def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    setup(seed=1234)
    args = parse_train_args()
    # initialize library
    if args.enbl_multi_gpu:
        from dglt.multi_gpu_wrapper import MultiGpuWrapper as mgw
        mgw.init()

    lg = RDLogger.logger()
    # Removing annoying warning during training.
    lg.setLevel(RDLogger.CRITICAL)
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    cross_validate(args, logger)
    if os.environ.get('CHIEF_IP', ''):
        job_completed()

