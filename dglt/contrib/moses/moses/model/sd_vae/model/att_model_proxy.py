#!/usr/bin/env python

from __future__ import print_function

# import os
# import sys
import numpy as np
from dglt.contrib.moses.moses.model.sd_vae.model.attribute_tree_decoder import create_tree_decoder
# from dglt.contrib.moses.moses.model.sd_vae.model.mol_decoder import batch_make_att_masks
from dglt.contrib.moses.moses.model.sd_vae.model.tree_walker import ConditionalDecoder

# from dglt.contrib.moses.moses.model.sd_vae.config import get_parser
from dglt.contrib.moses.moses.model.sd_vae.utils.mol_tree import get_smiles_from_tree, Node

def decode_single(pred_logits, utils, use_random, decode_times):
    tree_decoder = create_tree_decoder(utils)
    result = []
        
    walker = ConditionalDecoder(np.squeeze(pred_logits), use_random, utils)

    for _decode in range(decode_times):
        new_t = Node('smiles', utils.prod)
        try:
            tree_decoder.decode(new_t, walker)
            sampled = get_smiles_from_tree(new_t)
        except Exception as ex:
            import random, string
            if not type(ex).__name__ == 'DecodingLimitExceeded':
                print('Warning, decoder failed with %s' % ex)
                sampled = 'JUNK-ERR' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(256))
            else:
                # failed. output a random junk.
                sampled = 'JUNK-EX' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(256))

        result.append(sampled)

    return result

