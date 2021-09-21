#!/usr/bin/env python

import numpy as np

from dglt.contrib.moses.moses.model.sd_vae.utils import cfg_parser


class Node(object):
    def __init__(self,s, prod, father = None):
        self.symbol = s
        self.children = []
        self.rule_used = None    
        self.init_atts()
        self.father = father
        self.prod = prod
        if self.father is None:
            assert self.symbol == 'smiles'

    def init_atts(self):
        self.left_remain = None
        self.right_remain = None
        self.single_atom = None
        self.task = False
        self.pre_node = None
        self.atom_pos = None
        self.is_aromatic = None

    def is_created(self):
        if self.rule_used is None:
            return False
        return len(self.prod[self.symbol][self.rule_used]) == len(self.children)

    def add_child(self, child, pos = None):
        if self.is_created():
            return
        if pos is None:
            self.children.append(child)
        else:
            self.children.insert(pos, child)

    def get_pre(self):
        if self.pre_node is None:
            if self.father is None:
                self.pre_node = -1
            else:
                self.pre_node = self.father.get_pre()
        assert self.pre_node is not None
        return self.pre_node

def dfs(node, result):
    if len(node.children):
        for c in node.children:
            dfs(c, result)
    else:
        assert node.symbol[0] == node.symbol[-1] == '\''
        result.append(node.symbol[1:-1])

def get_smiles_from_tree(root):
    result = []
    dfs(root, result)
    st = ''.join(result)
    return st

def _AnnotatedTree2MolTree(annotated_root, bond_set, father, utils):
    n = Node(str(annotated_root.symbol), utils.prod, father=father)
    n.rule_used = annotated_root.rule_selection_id    
    for c in annotated_root.children:
        new_c = _AnnotatedTree2MolTree(c, bond_set, n, utils)
        n.children.append(new_c)
    if n.symbol == 'ringbond':
        assert len(n.children)
        d = n.children[-1]
        assert d.symbol == 'DIGIT'
        st = d.children[0].symbol
        assert len(st) == 3
        if len(n.children) > 2:
            d2 = n.children[-2]
            assert d2.symbol == 'DIGIT'
            st2 = d2.children[0].symbol
            assert len(st2) == 3
            idx = int(st2[1:-1]) * 10
        else:
            idx = 0
        idx += int(st[1 : -1]) - 1
        if idx in bond_set:
            n.bond_idx = idx
            bond_set.remove(idx)
        else:
            bond_set.add(idx)
            n.bond_idx = utils.MAX_NESTED_BONDS
    if isinstance(annotated_root.symbol, cfg_parser.Nonterminal): # it is a non-terminal        
        assert len(n.children)
        assert n.is_created()
    else:
        assert isinstance(annotated_root.symbol, str)
        assert len(n.symbol) < 3 or (n.symbol[0] != '\'' and n.symbol[-1] != '\'')        
        n.symbol = '\'' + n.symbol + '\''        
    return n

def AnnotatedTree2MolTree(annotated_root, utils):
    bond_set = set()
    ans = _AnnotatedTree2MolTree(annotated_root, bond_set, father=None, utils=utils)
    assert len(bond_set) == 0
    return ans

def dfs_indices(node, result, utils):
    if len(node.children):
        assert node.rule_selection_id >= 0
        g_range = utils.rule_ranges[str(node.symbol)]
        idx = g_range[0] + node.rule_selection_id
        assert idx >= 0 and idx < utils.TOTAL_NUM_RULES

        result.append(idx)
        for c in node.children:
            dfs_indices(c, result, utils)

def AnnotatedTree2RuleIndices(annotated_root, utils):
    result = []
    dfs_indices(annotated_root, result, utils)
    return np.array(result)

def AnnotatedTree2Onehot(annotated_root, max_len, utils):
    cur_indices = AnnotatedTree2RuleIndices(annotated_root, utils)
    assert len(cur_indices) <= max_len

    x_cpu = np.zeros(( utils.DECISION_DIM, max_len ), dtype=np.float32)
    x_cpu[cur_indices, np.arange(len(cur_indices))] = 1.0
    x_cpu[-1, np.arange(len(cur_indices), max_len)] = 1.0 # padding

    return x_cpu

if __name__ == '__main__':
    from dglt.contrib.moses.moses.model.sd_vae.config import get_parser

    cmd_args, _ = get_parser().parse_known_args()

    smiles = 'OSC'
    grammar = cfg_parser.Grammar(cmd_args.grammar_file)


    ts = cfg_parser.parse(smiles, grammar)
    assert isinstance(ts, list) and len(ts) == 1

    print(AnnotatedTree2RuleIndices(ts[0]))
