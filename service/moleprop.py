import os
import time
import math
import numpy as np
import torch

# torch.multiprocessing.set_start_method('spawn')
torch.multiprocessing.set_start_method('forkserver', force=True)
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import Namespace
from typing import List
from dglt.data.dataset.molecular import MoleculeDataset
from dglt.data.transformer.scaler import StandardScaler
from dglt.data.transformer.collator import MolCollator
from dglt.data.dataset.utils import get_data, get_data_from_smiles
from dglt.utils import load_args, load_checkpoint, load_scalers
from deploy import get_newest_train_args
from third_party.dimorphite_dl.acid_base import mol_cls

class MoleProp(object):
    """Molecular Properties Prediction Service"""

    def __init__(self, checkpoint_dir, debug=print):
        self.debug_ = debug
        self.checkpoint_paths_ = []
        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith('.pt'):
                    self.checkpoint_paths_.append(os.path.join(root, fname))

    def load_model(self, args: Namespace):
        """
        Load checkpoints

        :param args: Arguments.
        :return:
        """
        self.scaler_, self.features_scaler_ = load_scalers(self.checkpoint_paths_[0])
        self.train_args = load_args(self.checkpoint_paths_[0])
        self.args_ = args
        for key, value in vars(self.train_args).items():
            if not hasattr(self.args_, key):
                setattr(self.args_, key, value)

        # update args with newest training args
        newest_train_args = get_newest_train_args()
        for key, value in vars(newest_train_args).items():
            if not hasattr(args, key):
                setattr(args, key, value)
        if args.features_path:
            args.features_path = None
            args.features_generator = ['rdkit_2d_normalized']
        self.models_ = []
        for checkpoint_path in tqdm(self.checkpoint_paths_, total=len(self.checkpoint_paths_)):
            self.models_.append(load_checkpoint(checkpoint_path, cuda=self.args_.cuda, current_args=self.args_))

    def inference(self,
                  model: nn.Module,
                  data: MoleculeDataset,
                  args,
                  batch_size: int,
                  shared_dict,
                  scaler: StandardScaler = None
                  ) -> List[List[float]]:
        """
        Do inference
        :param model: model.
        :param data: input data.
        :param args: Arguments.
        :param batch_size: batch size.
        :param shared_dict: shared_dict of model.
        :param scaler: scaler of input data.
        :return: prediction of molecular properties.
        """
        # model.share_memory()
        model.eval()
        args.bond_drop_rate = 0
        preds = []
        iter_count = 0
        mol_collator = MolCollator(args=args, shared_dict=shared_dict)
        mol_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=mol_collator)
        for i, item in enumerate(mol_loader):
            smiles_batch, batch, features_batch, mask, _ = item

            with torch.no_grad():
                batch_preds = model(batch, features_batch)
                iter_count += args.batch_size
            batch_preds = batch_preds.data.cpu().numpy()
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
            batch_preds = batch_preds.tolist()
            preds.extend(batch_preds)

        return preds

    def postprocessing(self, task: str = None, smiles: List[str] = None, preds: np.ndarray = None):
        if task == 'caco2':
            for i in range(preds.shape[0]):
                if preds[i] is not None:
                    for j in range(len(preds[i])):
                        preds[i][j] = (math.pow(10, preds[i][j]) - 1) / 10
        elif task == 'pka':
            acid_base = mol_cls(smiles)
            preds[acid_base == None] = np.nan
            preds = np.column_stack((preds, np.array(acid_base, dtype=np.float)))
        elif task == 'ppb':
            preds[preds > 1] = 1
            preds[preds < 0] = 0
        return preds

    def predict(self, task: str = None, smiles: List[str] = None):
        """
        Predict molecular properties.
        :param smiles: input data.
        :return: molecular properties.
        """
        self.debug_('Loading data')
        tic = time.time()
        self.args_.max_workers = 30
        if smiles is not None:
            test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=True, args=self.args_)
        else:
            test_data = get_data(path=self.args_.input_file, args=self.args_,
                                 use_compound_names=self.args_.use_compound_names,
                                 skip_invalid_smiles=True)
        toc = time.time()
        self.debug_('loading data: {}s'.format(toc - tic))
        self.debug_('Validating SMILES')
        tic = time.time()
        valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
        full_data = test_data
        test_data = MoleculeDataset([test_data[i] for i in valid_indices])

        # Edge case if empty list of smiles is provided
        if len(test_data) == 0:
            return [None] * len(full_data)

        # Normalize features
        if self.train_args.features_scaling:
            test_data.normalize_features(self.features_scaler)

        sum_preds = np.zeros((len(test_data), self.args_.num_tasks))
        toc = time.time()
        self.debug_('validating smiles: {}s'.format(toc - tic))
        self.debug_(f'Predicting...')
        tic = time.time()
        shared_dict = {}
        for model in self.models_:
            model_preds = self.inference(
                model=model,
                data=test_data,
                batch_size=self.args_.batch_size,
                scaler=self.scaler_,
                shared_dict=shared_dict,
                args=self.args_
            )
            sum_preds += np.array(model_preds)
        toc = time.time()
        self.debug_('predicting: {}s'.format(toc - tic))
        avg_preds = sum_preds / len(self.checkpoint_paths_)
        avg_preds = self.postprocessing(task=task, smiles=smiles, preds=avg_preds)
        avg_preds = avg_preds.tolist()
        assert len(test_data) == len(avg_preds)
        test_smiles = test_data.smiles()
        res = {}
        for i in range(len(avg_preds)):
            res[test_smiles[i]] = avg_preds[i]

        return {'task': task, 'task_score': res}
