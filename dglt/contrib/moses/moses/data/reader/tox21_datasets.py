"""
Tox21 dataset loader.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import logging
import deepchem

import dglt.contrib.moses.moses.data.featurizer as feat
from dglt.contrib.moses.moses.data.reader.utils import AdditionalInfo
from dglt.contrib.moses.moses.data.reader.data_loader import CSVLoader

logger = logging.getLogger(__name__)

def get_tox21_data_path():
    data_dir = deepchem.utils.get_data_dir()
    dataset_file = os.path.join(data_dir, "tox21.csv.gz")
    if not os.path.exists(dataset_file):
        deepchem.utils.download_url(
            'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
        )
    return dataset_file

dataset_file = get_tox21_data_path()
tox21_ad = AdditionalInfo(dataset_file, smiles_field='smiles')

def load_tox21(featurizer='ECFP', split='index', reload=True, K=1):
    """Load Tox21 datasets. Does not do train/test split"""
    # Featurize Tox21 dataset
    tox21_tasks = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]

    data_dir = deepchem.utils.get_data_dir()
    # TODO: reload should be modified to support cross vailidation cases.
    if reload and K == 1:
        save_dir = os.path.join(data_dir, "tox21/" + featurizer + "/" + str(split))
        loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
            save_dir)
        if loaded:
            return tox21_tasks, all_dataset, transformers

    dataset_file = get_tox21_data_path()

    if featurizer == 'ECFP':
        featurizer = feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
        featurizer = feat.ConvMolFeaturizer()
    elif featurizer == 'Weave':
        featurizer = feat.WeaveFeaturizer()
    elif featurizer == 'Raw':
        featurizer = feat.RawFeaturizer()
    elif featurizer == 'AdjacencyConv':
        featurizer = feat.AdjacencyFingerprint(
            max_n_atoms=150, max_valence=6)
    elif featurizer == 'EAGCN':
        featurizer = feat.EagcnFeaturizer(tox21_ad.bond_type_dict, tox21_ad.atom_type_dict)

    loader = CSVLoader(
        tasks=tox21_tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(dataset_file, shard_size=8192)

    # Initialize transformers
    transformers = [
        deepchem.trans.BalancingTransformer(transform_w=True, dataset=dataset)
    ]

    logger.info("About to transform data")
    for transformer in transformers:
        dataset = transformer.transform(dataset)

    if split == None:
        return tox21_tasks, (dataset, None, None), transformers

    splitters = {
        'index': deepchem.splits.IndexSplitter(),
        'random': deepchem.splits.RandomSplitter(),
        'scaffold': deepchem.splits.ScaffoldSplitter(),
        # 'butina': deepchem.splits.ButinaSplitter(),
        # 'task': deepchem.splits.TaskSplitter()
    }
    splitter = splitters[split]

    if K > 1:
        fold_datasets = splitter.k_fold_split(dataset, K)
        all_dataset = fold_datasets
    else:
        train, valid, test = splitter.train_valid_test_split(dataset)
        all_dataset = (train, valid, test)
        if reload:
            deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                                     transformers)
    return tox21_tasks, all_dataset, transformers
