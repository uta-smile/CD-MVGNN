#!/usr/bin/env python

import os
from collections import defaultdict
import nltk


class MolUtil():
    def __init__(self, cmd_args):
        if "info_folder" not in cmd_args or cmd_args.info_folder is None:
            info_folder = os.path.dirname(cmd_args.grammar_file)
        else:
            info_folder = cmd_args.info_folder

        prod = defaultdict(list)

        _total_num_rules = 0
        rule_ranges = {}
        terminal_idxes = {}

        avail_atoms = {}
        aliphatic_types = []
        aromatic_types = []
        element_types = []
        aromatic_symbol_types = []
        bond_types = []

#        cfg_string = ''.join(list(open(cmd_args.grammar_file).readlines()))
#        cfg_grammar = nltk.CFG.fromstring(cfg_string)
#
#        for p in cfg_grammar.productions():
#            head = p.lhs().symbol()
#            rule = p.rhs()
#            prod[head].append(tuple([_.symbol() if isinstance(_, nltk.Nonterminal)
#                                     else "\'" + _ + "\'" for _ in rule]))
#            for x in rule:
#                if isinstance(x, str) and not x in terminal_idxes:
#                    idx = len(terminal_idxes)
#                    terminal_idxes["\'" + x + "\'"] = idx
#
#        for s, rules in prod.items():
#            rule_ranges[s] = (_total_num_rules, _total_num_rules + len(rules))
#            _total_num_rules += len(rules)
#
#            if s == 'aliphatic_organic':
#                for x in rules:
#                    assert len(x) == 1
#                    aliphatic_types.append(x[0])
#            if s == 'aromatic_organic':
#                for x in rules:
#                    assert len(x) == 1
#                    aromatic_types.append(x[0])
#            if s == 'element_symbols':
#                for x in rules:
#                    assert len(x) == 1
#                    element_types.append(x[0])
#            if s == 'aromatic_symbols':
#                for x in rules:
#                    assert len(x) == 1
#                    aromatic_symbol_types.append(x[0])
#            if s == 'bond':
#                for x in rules:
#                    assert len(x) == 1
#                    bond_types.append(x[0])

        with open(cmd_args.grammar_file, 'r') as f:
            for row in f:
                s = row.split('->')[0].strip()
                rules = row.split('->')[1].strip().split('|')
                rules = [w.strip() for w in rules]
                for rule in rules:
                    rr = rule.split()
                    prod[s].append(rr)
                    for x in rr:
                        if x[0] == '\'' and not x in terminal_idxes:
                            idx = len(terminal_idxes)
                            terminal_idxes[x] = idx
                rule_ranges[s] = (_total_num_rules, _total_num_rules + len(rules))
                _total_num_rules += len(rules)
        
                if s == 'aliphatic_organic':
                    for x in prod[s]:
                        assert len(x) == 1
                        aliphatic_types.append(x[0])
                if s == 'aromatic_organic':
                    for x in prod[s]:
                        assert len(x) == 1
                        aromatic_types.append(x[0])
                if s == 'aromatic_symbols':
                    for x in prod[s]:
                        assert len(x) == 1
                        aromatic_symbol_types.append(x[0])
                if s == 'element_symbols':
                    for x in prod[s]:
                        assert len(x) == 1
                        element_types.append(x[0])
                if s == 'bond':
                    for x in prod[s]:
                        assert len(x) == 1
                        bond_types.append(x[0])
        
        def load_valence(fname, info_dict):
            with open(fname, 'r') as f:
                for row in f:
                    row = row.split()
                    info_dict[row[0]] = int(row[1])

        avail_atoms['aliphatic_organic'] = aliphatic_types
        avail_atoms['aromatic_organic'] = aromatic_types
        avail_atoms['element_symbols'] = element_types
        avail_atoms['aromatic_symbols'] = aromatic_symbol_types
        TOTAL_NUM_RULES = _total_num_rules
        atom_valence = {}
        bond_valence = {}
        load_valence(info_folder + '/atom.valence', atom_valence)
        load_valence(info_folder + '/bond.valence', bond_valence)
        bond_valence[None] = 1
        MAX_NESTED_BONDS = 12

        DECISION_DIM = MAX_NESTED_BONDS + TOTAL_NUM_RULES + 2

        self.DECISION_DIM = DECISION_DIM
        self.TOTAL_NUM_RULES = TOTAL_NUM_RULES
        self.MAX_NESTED_BONDS = MAX_NESTED_BONDS
        self.terminal_idxes = terminal_idxes
        self.avail_atoms = avail_atoms
        self.rule_ranges = rule_ranges
        self.atom_valence = atom_valence
        self.bond_valence = bond_valence
        self.bond_types = bond_types
        self.prod = prod

if __name__ == '__main__':
    from dglt.contrib.moses.moses.model.sd_vae.config import get_parser
    cmd_args, _ = get_parser().parse_known_args()

    print(MolUtil(cmd_args).terminal_idxes)
