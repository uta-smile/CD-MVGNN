#!/usr/bin/env python


from __future__ import print_function

import numpy as np


class DecodingLimitExceeded(Exception):
    def __init__(self):
        pass
    def __str__(self):
        return 'DecodingLimitExceeded'

class TreeWalker(object):

    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    def sample_index_with_mask(self, node, idxes):
        raise NotImplementedError

class OnehotBuilder(TreeWalker):

    def __init__(self, utils):
        super(OnehotBuilder, self).__init__()
        self.reset()
        self.TOTAL_NUM_RULES = utils.TOTAL_NUM_RULES
        self.rule_ranges = utils.rule_ranges
        
    def reset(self):
        self.num_steps = 0
        self.global_rule_used = []
        self.mask_list = []

    def sample_index_with_mask(self, node, idxes):
        assert node.rule_used is not None
        g_range = self.rule_ranges[node.symbol]
        global_idx = g_range[0] + node.rule_used
        self.global_rule_used.append(global_idx)
        self.mask_list.append(np.array(idxes))

        self.num_steps += 1

        result = None
        for i in range(len(idxes)):
            if idxes[i] == global_idx:
                result = i
        assert result is not None
        return result

    def sample_att(self, node, candidates):
        assert hasattr(node, 'bond_idx')
        assert node.bond_idx in candidates

        global_idx = self.TOTAL_NUM_RULES + node.bond_idx
        self.global_rule_used.append(global_idx)
        self.mask_list.append(np.array(candidates) + self.TOTAL_NUM_RULES)

        self.num_steps += 1
        
        return node.bond_idx

class ConditionalDecoder(TreeWalker):

    def __init__(self, raw_logits, use_random, utils):
        super(ConditionalDecoder, self).__init__()
        self.raw_logits = raw_logits
        self.use_random = use_random
        assert len(raw_logits.shape) == 2 and raw_logits.shape[1] == utils.DECISION_DIM
        self.TOTAL_NUM_RULES = utils.TOTAL_NUM_RULES

        self.reset()

    def reset(self):
        self.num_steps = 0

    def _get_idx(self, cur_logits):
        if self.use_random:
            cur_prob = np.exp(cur_logits)
            cur_prob = cur_prob / np.sum(cur_prob)

            result = np.random.choice(len(cur_prob), 1, p=cur_prob)[0]
            result = int(result)  # enusre it's converted to int
        else:
            result = np.argmax(cur_logits)

        self.num_steps += 1
        return result

    def sample_index_with_mask(self, node, idxes):
        if self.num_steps >= self.raw_logits.shape[0]:
            raise DecodingLimitExceeded()
        cur_logits = self.raw_logits[self.num_steps][idxes]

        self.cur_logits = np.zeros(self.raw_logits[self.num_steps].shape)
        self.cur_logits[idxes] = cur_logits
        cur_idx = self._get_idx(cur_logits)
        self.cur_idx = idxes[cur_idx]
        
        return cur_idx

    def sample_att(self, node, candidates):        
        if self.num_steps >= self.raw_logits.shape[0]:
            raise DecodingLimitExceeded()
        cur_logits = self.raw_logits[self.num_steps][np.array(candidates) + self.TOTAL_NUM_RULES]

        self.cur_logits = np.zeros(self.raw_logits[self.num_steps].shape)
        self.cur_logits[np.array(candidates) + self.TOTAL_NUM_RULES] = cur_logits
        cur_idx = candidates[self._get_idx(cur_logits)]
        self.cur_idx = cur_idx + self.TOTAL_NUM_RULES
        
        return cur_idx


class YieldDecoder(TreeWalker):
    def __init__(self, use_random, utils):
        super(YieldDecoder, self).__init__()
        self.use_random = use_random
        self.TOTAL_NUM_RULES = utils.TOTAL_NUM_RULES
        self.raw_logits = None

        self.reset()

    def reset(self):
        self.num_steps = 0

    def _get_idx(self, cur_logits):
        if self.use_random:
            cur_prob = np.exp(cur_logits)
            cur_prob = cur_prob / np.sum(cur_prob)

            result = np.random.choice(len(cur_prob), 1, p=cur_prob)[0]
            result = int(result)  # enusre it's converted to int
        else:
            result = np.argmax(cur_logits)

        self.num_steps += 1
        return result

    def sample_index_with_mask(self, node, idxes):
        # if self.num_steps >= self.raw_logits.shape[0]:
        #     raise DecodingLimitExceeded()
        cur_logits = self.raw_logits[idxes]

        self.cur_logits = np.zeros(self.raw_logits.shape)
        self.cur_logits[idxes] = cur_logits
        cur_idx = self._get_idx(cur_logits)
        self.cur_idx = idxes[cur_idx]

        return cur_idx

    def sample_att(self, node, candidates):
        # if self.num_steps >= self.raw_logits.shape[0]:
        #     raise DecodingLimitExceeded()
        cur_logits = self.raw_logits[np.array(candidates) + self.TOTAL_NUM_RULES]

        self.cur_logits = np.zeros(self.raw_logits.shape)
        self.cur_logits[np.array(candidates) + self.TOTAL_NUM_RULES] = cur_logits
        cur_idx = candidates[self._get_idx(cur_logits)]
        self.cur_idx = cur_idx + self.TOTAL_NUM_RULES

        return cur_idx


if __name__ == '__main__':
    pass

    
    
