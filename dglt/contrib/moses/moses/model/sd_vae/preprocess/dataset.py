import numpy as np
from collections import Counter

from dglt.contrib.moses.moses.model.sd_vae.model.att_model_proxy import decode_single
from dglt.contrib.moses.moses.model.sd_vae.model.attribute_tree_decoder import create_tree_decoder
from dglt.contrib.moses.moses.model.sd_vae.model.mol_decoder import batch_make_att_masks_single
from dglt.contrib.moses.moses.model.sd_vae.model.tree_walker import OnehotBuilder
from dglt.contrib.moses.moses.model.sd_vae.preprocess.csv_saver import line_reader
from dglt.contrib.moses.moses.model.sd_vae.utils import cfg_parser as parser
from dglt.contrib.moses.moses.model.sd_vae.utils.mol_util import MolUtil
from torch.utils.data import Dataset

from dglt.contrib.moses.moses.model.sd_vae.utils.mol_tree import AnnotatedTree2MolTree


class SDVAEDataset(Dataset):
    def __init__(self, data, cmd_args, smiles=None, grammar=None, onehot=True):
        self.data = data
        self.smiles = smiles
        assert self.smiles is None or len(self.data) == len(self.smiles), \
            "data and smiles does not match."
        if len(self.data[0].split()) > 1:
            self.preprocessed = True
        else:
            self.preprocessed = False

        self.onehot = onehot
        self.utils = MolUtil(cmd_args)
        if grammar is None:
            self.grammar = parser.Grammar(cmd_args.grammar_file)
        else:
            self.grammar = grammar

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.smiles is not None:
            smiles = self.smiles[index]
        else:
            smiles = None

        if self.preprocessed:
            onehot, masks = line_reader(self.data[index], self.utils, onehot=self.onehot)
        else:
            walker = OnehotBuilder(self.utils)
            tree_decoder = create_tree_decoder(self.utils)
            ts = parser.parse(self.data[index], self.grammar)
            assert isinstance(ts, list) and len(ts) == 1

            cfg_tree = AnnotatedTree2MolTree(ts[0], self.utils)

            onehot, masks = batch_make_att_masks_single(cfg_tree, self.utils, tree_decoder, walker,
                                                        dtype=float, onehot=self.onehot)

        return (onehot, masks, smiles)

    @property
    def is_onehot(self):
        return self.onehot

class SDVAEGenerater(Dataset):
    def __init__(self, raw_logits, utils, use_random=True, valid=True, sample_times=100):
        self.raw_logits = raw_logits
        self.utils = utils
        self.use_random = use_random
        self.sample_times = sample_times
        self.valid = valid

    def __len__(self):
        return self.raw_logits.shape[0]

    def __getitem__(self, index):
        pred_logits = self.raw_logits[index, :, :]
        results = decode_single(pred_logits, self.utils,
                               use_random=self.use_random,
                               decode_times=self.sample_times)

        cnt = Counter()
        for result in results:
            if not result.startswith('JUNK'):
                cnt[result] += 1
        # print(cnt)
        if len(cnt) > 0:
            decoded = cnt.most_common(1)[0][0]
        else:
            if self.valid:
                return None
            decoded = results[0]
        return decoded
