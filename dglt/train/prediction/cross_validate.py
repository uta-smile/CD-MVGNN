import os
import time
from argparse import Namespace
from logging import Logger
from typing import Tuple

import numpy as np

from dglt.data.dataset.utils import get_task_names
from dglt.multi_gpu_wrapper import MultiGpuWrapper as mgw
from dglt.utils import makedirs
from .run_training import run_training


def cross_validate(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    master_worker = (mgw.rank() == 0) if args.enbl_multi_gpu else True
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    # Run training on different random seeds for each fold
    all_scores = []
    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        model_scores = run_training(args, time_start, logger)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report scores for each fold
    if master_worker:
        # Report results
        info(f'{args.num_folds}-fold cross validation')

        for fold_num, scores in enumerate(all_scores):
            info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

            if args.show_individual_scores:
                for task_name, score in zip(task_names, scores):
                    info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

        # Report scores across models
        avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
        mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
        info(f'overall_{args.split_type}_test_{args.metric}={mean_score:.6f}')
        info(f'std={std_score:.6f}')

        if args.show_individual_scores:
            for task_num, task_name in enumerate(task_names):
                info(f'Overall test {task_name} {args.metric} = '
                     f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

        return mean_score, std_score
