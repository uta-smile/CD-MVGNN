"""
bace dataset loader.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import logging
import deepchem
from deepchem.molnet.load_function.bace_features import bace_user_specified_features

import dglt.contrib.moses.moses.data.featurizer as feat
from dglt.contrib.moses.moses.data.reader.utils import AdditionalInfo
from dglt.contrib.moses.moses.data.reader.data_loader import CSVLoader


logger = logging.getLogger(__name__)


def get_bace_data_path():
    data_dir = deepchem.utils.get_data_dir()
    dataset_file = os.path.join(data_dir, "bace.csv")

    if not os.path.exists(dataset_file):
        deepchem.utils.download_url(
            'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/bace.csv'
        )
    return dataset_file

dataset_file = get_bace_data_path()
bace_ad = AdditionalInfo(dataset_file, smiles_field="mol")

def load_bace_classification(featurizer='ECFP', split='random', reload=True):
    """Load bace datasets."""
    # Featurize bace dataset
    logger.info("About to featurize bace dataset.")
    data_dir = deepchem.utils.get_data_dir()
    dataset_file = get_bace_data_path()
    if reload:
        save_dir = os.path.join(data_dir, "bace_c/" + featurizer + "/" + str(split))
    bace_tasks = ["Class"]
    if reload:
        loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
            save_dir)
        if loaded:
            return bace_tasks, all_dataset, transformers

    if featurizer == 'ECFP':
        featurizer = deepchem.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
        featurizer = deepchem.feat.ConvMolFeaturizer()
    elif featurizer == 'Weave':
        featurizer = deepchem.feat.WeaveFeaturizer()
    elif featurizer == 'Raw':
        featurizer = deepchem.feat.RawFeaturizer()
    elif featurizer == 'UserDefined':
        featurizer = deepchem.feat.UserDefinedFeaturizer(
            bace_user_specified_features)
    elif featurizer == 'EAGCN':
        featurizer = feat.EagcnFeaturizer(bace_ad.bond_type_dict, bace_ad.atom_type_dict)

    loader = CSVLoader(
        tasks=bace_tasks, smiles_field="mol", featurizer=featurizer)

    dataset = loader.featurize(dataset_file, shard_size=8192)
    # Initialize transformers
    transformers = [
        deepchem.trans.BalancingTransformer(transform_w=True, dataset=dataset)
    ]

    logger.info("About to transform data")
    for transformer in transformers:
        dataset = transformer.transform(dataset)

    if split == None:
        return bace_tasks, (dataset, None, None), transformers

    splitters = {
        'index': deepchem.splits.IndexSplitter(),
        'random': deepchem.splits.RandomSplitter(),
        'scaffold': deepchem.splits.ScaffoldSplitter()
    }
    splitter = splitters[split]
    train, valid, test = splitter.train_valid_test_split(dataset)

    if reload:
        deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                                 transformers)
    return bace_tasks, (train, valid, test), transformers
