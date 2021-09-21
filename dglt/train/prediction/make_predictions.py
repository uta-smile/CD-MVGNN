from argparse import Namespace
from typing import List, Optional
from io import BytesIO

import numpy as np
import pandas as pd
from tqdm import tqdm as core_tqdm

from dglt.data.dataset.molecular import MoleculeDataset
from dglt.data.dataset.utils import get_data, get_data_from_smiles
from dglt.utils import create_logger
from dglt.utils import load_args, load_checkpoint, load_scalers
from .predict import predict


class tqdm(core_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ascii", True)
        super(tqdm, self).__init__(*args, **kwargs)


def make_predictions(args: Namespace, newest_train_args=None, smiles: List[str] = None) -> List[Optional[List[float]]]:
    """
    Makes predictions. If smiles is provided, makes predictions on smiles. Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :param smiles: Smiles to make predictions on.
    :return: A list of lists of target predictions.
    """
    # if args.gpu is not None:
    #     torch.cuda.set_device(args.gpu)

    print('Loading training args')
    if hasattr(args, 'deploy'):
        from decrypt import decrypt_file
        path = decrypt_file(args.checkpoint_paths[0])
        path = BytesIO(path)
        scaler, features_scaler = load_scalers(path)
        path.seek(0)
        train_args = load_args(path)
    else:
        path = args.checkpoint_paths[0]
        scaler, features_scaler = load_scalers(path)
        train_args = load_args(path)

    # Update args with training arguments saved in checkpoint
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # update args with newest training args
    if newest_train_args is not None:
        for key, value in vars(newest_train_args).items():
            if not hasattr(args, key):
                setattr(args, key, value)

    if hasattr(args, 'deploy'):
        if args.features_path:
            args.features_path = None
            args.features_generator = ['rdkit_2d_normalized']
            args.features_scaling = False

    # deal with multiprocess problem
    args.debug = True

    logger = create_logger('predict', quiet=True)
    print('Loading data')
    if smiles is not None:
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False)
    else:
        test_data = get_data(path=args.test_path, args=args, use_compound_names=args.use_compound_names, skip_invalid_smiles=False)

    print('Validating SMILES')
    valid_indices = [i for i in range(len(test_data))]
    full_data = test_data
    test_data = MoleculeDataset([test_data[i] for i in valid_indices])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    if args.use_compound_names:
        compound_names = test_data.compound_names()
    print(f'Test size = {len(test_data):,}')

    # Normalize features
    if train_args.features_scaling:
        test_data.normalize_features(features_scaler)

    # Predict with each model individually and sum predictions
    sum_preds = np.zeros((len(test_data), args.num_tasks))
    print(f'Predicting...')
    shared_dict = {}
    # loss_func = torch.nn.BCEWithLogitsLoss()
    for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
        # Load model
        if hasattr(args, 'deploy'):
            from decrypt import decrypt_file
            checkpoint_path = decrypt_file(checkpoint_path)
            checkpoint_path = BytesIO(checkpoint_path)
            checkpoint_path.seek(0)
        model = load_checkpoint(checkpoint_path, cuda=args.cuda, current_args=args, logger=logger)
        model_preds, _ = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler,
            shared_dict=shared_dict,
            args=args,
            logger=logger,
            loss_func=None
        )
        sum_preds += np.array(model_preds, dtype=float)

    # Ensemble predictions
    avg_preds = sum_preds / len(args.checkpoint_paths)
    avg_preds = avg_preds.tolist()

    # Save predictions
    assert len(test_data) == len(avg_preds)

    # Put Nones for invalid smiles
    full_preds = [None] * len(full_data)
    for i, si in enumerate(valid_indices):
        full_preds[si] = avg_preds[i]
    avg_preds = full_preds
    test_smiles = full_data.smiles()
    avg_preds = np.array(avg_preds)
    return avg_preds, test_smiles


def write_prediction(avg_preds, test_smiles, args):
    result = pd.DataFrame(data=avg_preds, index=test_smiles, columns=args.task_names)
    result.to_csv(args.preds_path)
    print(f'Saving predictions to {args.preds_path}')
