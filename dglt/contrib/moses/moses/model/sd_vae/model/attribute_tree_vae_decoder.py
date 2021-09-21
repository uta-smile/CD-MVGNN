#!/usr/bin/env python


from __future__ import print_function

from collections import defaultdict

import numpy as np
from dglt.contrib.moses.moses.model.sd_vae.config import get_parser
from dglt.contrib.moses.moses.model.sd_vae.model.tree_walker import YieldDecoder

from dglt.contrib.moses.moses.model.sd_vae.utils.mol_tree import Node

cmd_args, _ = get_parser().parse_known_args()


class RingBond(object):
    def __init__(self, pos, b_type):
        self.pos = pos
        self.b_type = b_type


class AttMolGraphDecoders(object):
    def __init__(self, len, utils, use_random=False, n_workers=0):
        self.decoders = []
        self.nodes = []
        self.num_workers = n_workers
        self.len = len
        self.eos_mark = np.zeros(len, dtype=bool)
        self.junks = np.zeros(len, dtype=bool)
        for _ in range(len):
            walker = YieldDecoder(use_random, utils)
            self.nodes.append(Node('smiles', utils.prod))
            self.decoders.append(AttMolGraphDecoder(utils))
            self.decoders[-1].decode(self.nodes[-1], walker)

    def getCandidate(self, raw_logits, num_step, pad):
        assert raw_logits.shape[0] == self.len

        y = []
        num_step += 1
        for idx, (logit, decoder) in enumerate(zip(raw_logits, self.decoders)):
            if not self.eos_mark[idx] and not self.junks[idx]:
                decoder.walker.raw_logits = logit
                try:
                    while decoder.walker.num_steps < num_step:
                        next(decoder.generator)
                    assert decoder.walker.num_steps == num_step, "step is not matched."
                    y.append(decoder.walker.cur_idx)
                except StopIteration:
                    self.eos_mark[idx] = True
                    y.append(pad)
                except AssertionError as ex:
                    print('Warning, decoder failed with %s' % ex)
                    self.junks[idx] = True
                    self.eos_mark[idx] = True
                    y.append(pad)
            else:
                y.append(pad)
            if np.all(self.eos_mark):
                break
        return y


class AttMolGraphDecoder(object):
    def __init__(self, utils):
        self.reset_state()
        self.rule_ranges = utils.rule_ranges
        self.avail_atoms = utils.avail_atoms
        self.atom_valence = utils.atom_valence
        self.bond_types = utils.bond_types
        self.bond_valence = utils.bond_valence
        self.prod = utils.prod
        self.MAX_NESTED_BONDS = utils.MAX_NESTED_BONDS

    def reset_state(self):
        self.atom_num = 0
        self.matched_bonds = set()
        self.open_rings = {}
        self.sameatom_bonds = defaultdict(set)

    def get_node(self, node, new_sym, pos):
        if node.is_created():
            assert pos < len(node.children)
            ans = node.children[pos]
            ans.init_atts()
            assert ans.symbol == new_sym
            return ans
        return Node(new_sym, self.prod, node)

    def rand_rule(self, node, sub_ranges = None):
        g_range = self.rule_ranges[node.symbol]
        idxes = np.arange(g_range[0], g_range[1])
        if sub_ranges is not None:
            idxes = idxes[sub_ranges]

        assert len(idxes)
        if len(idxes) == 1 and cmd_args.skip_deter:
            result = 0
            self.walker.num_steps += 1
        else:
            result = self.walker.sample_index_with_mask(node, idxes)

        if sub_ranges is not None:
            new_idx = sub_ranges[result]
        else:
            new_idx = result
        
        if node.rule_used is not None:
            assert node.rule_used == new_idx
        else:
            node.rule_used = new_idx

        return node.rule_used

    def rand_att(self, node, candidates):
        if len(candidates) == 1 and cmd_args.skip_deter:
            att_idx = candidates[0]
            self.walker.num_steps += 1
        else:
            att_idx = self.walker.sample_att(node, candidates)
        if not hasattr(node, 'bond_idx'):
            node.bond_idx = att_idx
        else:
            assert node.bond_idx == att_idx
        return att_idx

    def ring_valid(self, r, pre_pos, remain):
        p = (self.open_rings[r].pos, self.atom_num - 1)
        if self.open_rings[r].pos == self.atom_num - 1:
            return False
        if self.open_rings[r].pos == pre_pos:
            return False
        if p in self.matched_bonds:
            return False
        if self.bond_valence[self.open_rings[r].b_type] > remain:
            return False
        return True

    def maximum_match(self, pre_pos, remain):
        if remain == 0:
            return 0
        cur_pos = self.atom_num - 1
        s = set()
        ans = 0
        rest = remain
        for cost in range(1, 4):
            for r in self.open_rings:
                if self.bond_valence[self.open_rings[r].b_type] != cost:
                    continue
                if self.ring_valid(r, pre_pos, rest) and not self.open_rings[r].pos in s:
                    s.add(self.open_rings[r].pos)
                    rest -= 1
                    ans += 1
                    assert rest >= 0
                    if rest == 0:
                        return ans
        return ans
            
    def tree_generator(self, node, left_conn = False, right_conn = False, cap_remain = None, ref_symbol = None, is_last = None):
        assert is_last is not None
        if node.symbol in ['bond', 'BB', 'branch', 'BAC', 'BAH', 'charge', 'hcount']:
            assert cap_remain is not None
        
        if node.symbol == 'chain':
            rule = self.rand_rule(node)
            yield
            a = self.get_node(node, 'branched_atom', 0)
            node.add_child(a)
            if rule == 0: # chain -> branched_atom
                yield from self.tree_generator(a, left_conn, right_conn, is_last = is_last)
                node.left_remain = a.left_remain
                node.right_remain = a.right_remain
                node.single_atom = True
            else:
                yield from self.tree_generator(a, left_conn, True, is_last = False)
                c = self.get_node(node, 'chain', -1)
                c.pre_node = a.atom_pos
                assert c.pre_node is not None
                yield from self.tree_generator(c, True, right_conn, is_last = is_last)
                
                cost = 0
                if rule == 2: # chain -> chain bond branched_atom
                    b = self.get_node(node, 'bond', 1)
                    yield from self.tree_generator(b, cap_remain = min(c.left_remain, a.right_remain) + 1, is_last=is_last)
                    cost = self.bond_valence[b.children[0].symbol] - 1
                    node.add_child(b)
                if rule == 3: # chain -> branched_atom '.' chain
                    b = self.get_node(node, '\'.\'', 1)
                    node.add_child(b)
                node.add_child(c)
                node.left_remain = a.left_remain - cost
                node.right_remain = c.right_remain
                if c.single_atom:
                    node.right_remain = c.right_remain - cost
                assert node.left_remain >= 0
                assert node.right_remain >= 0
                node.single_atom = False
        elif node.symbol == 'aliphatic_organic' or node.symbol == 'aromatic_organic' \
                or node.symbol == 'element_symbols' or node.symbol == 'aromatic_symbols':
            min_valence = int(left_conn) + int(right_conn)
            if len(self.open_rings) and is_last:
                min_valence += 1
            candidates = []
            atom_types = self.avail_atoms[node.symbol]
            for i in range(len(atom_types)):
                a = atom_types[i]
                if self.atom_valence[a] >= min_valence:
                    if hasattr(node, 'banned_set') and a in node.banned_set:
                        continue
                    candidates.append(i)
            rule = self.rand_rule(node, candidates)
            yield
            a = self.get_node(node, atom_types[rule], 0)
            assert self.atom_valence[a.symbol] >= min_valence
            node.add_child(a)
            node.left_remain = self.atom_valence[a.symbol] - min_valence
            node.right_remain = self.atom_valence[a.symbol] - min_valence
            node.single_atom = True
            node.atom_pos = self.atom_num
            if node.symbol == 'aromatic_organic' or node.symbol == 'aromatic_symbols':
                node.is_aromatic = True
            else:
                node.is_aromatic = False
            self.atom_num += 1
        elif node.symbol == 'bond':
            candidates = []
            assert cap_remain            
            rr = range(len(self.bond_types))
            if hasattr(node, 'allowed'):
                rr = node.allowed
            for i in rr:
                b = self.bond_types[i]
                if self.bond_valence[b] <= cap_remain:
                    candidates.append(i)
            rule = self.rand_rule(node, candidates)
            yield
            b = self.get_node(node, self.bond_types[rule], 0)
            node.add_child(b)
        elif node.symbol == 'branched_atom':
            a = self.get_node(node, 'atom', 0)
            yield from self.tree_generator(a, left_conn, right_conn, is_last=is_last)
            node.atom_pos = a.atom_pos
            node.is_aromatic = a.is_aromatic
            node.add_child(a)

            candidates = set([0, 1, 2, 3])
            remain = int(a.left_remain)
            
            if len(self.open_rings) and is_last:
                remain += 1
                candidates.remove(0)
                pre_idx = node.get_pre()
                if self.maximum_match(pre_idx, remain) < len(self.open_rings):
                    candidates.remove(2)
                if remain < 2:
                    candidates.remove(3)
            else:
                if remain < 2:
                    candidates.remove(3)
                if remain < 1:
                    candidates.remove(2)
                    candidates.remove(1)
                if len(self.open_rings) == 0 and is_last:
                    if 2 in candidates:
                        candidates.remove(2)
                pre_idx = node.get_pre()
                if self.maximum_match(pre_idx, remain) == 0 and len(self.open_rings) == self.MAX_NESTED_BONDS:
                    assert not is_last
                    if 2 in candidates:
                        candidates.remove(2)
                    if 3 in candidates:
                        candidates.remove(3)
                if self.maximum_match(pre_idx, remain - 1) == 0 and len(self.open_rings) == self.MAX_NESTED_BONDS:
                    assert not is_last
                    if 3 in candidates:
                        candidates.remove(3)

            rule = self.rand_rule(node, list(candidates))
            yield

            if rule > 1: # branched_atom -> atom RB | atom RB BB                        
                r = self.get_node(node, 'RB', 1)
                if rule == 2 and is_last:
                    r.task = True
                remain = yield from self.tree_generator(r, cap_remain=remain - (rule == 3), is_last=is_last)
                remain += (rule == 3)
                node.add_child(r)
                        
            node.left_remain = remain
            if rule % 2 == 1: # branched_atom -> atom BB | atom RB BB
                assert remain > 0
                b = self.get_node(node, 'BB', -1)
                b.pre_node = a.atom_pos
                assert b.pre_node is not None
                node.left_remain = yield from self.tree_generator(b, cap_remain = remain, is_last=is_last)
                node.add_child(b)

            node.right_remain = node.left_remain
            node.single_atom = True
        elif node.symbol == 'RB':
            assert cap_remain
            b = self.get_node(node, 'ringbond', 0)
            b.task = node.task
            cap_remain = yield from self.tree_generator(b, cap_remain=cap_remain, is_last=is_last)
            node.add_child(b)
            
            candidates = []
            if node.task:
                candidates = [ int(len(self.open_rings) > 0) ]
            else:
                candidates = [0]
                pre_idx = node.get_pre()
                if cap_remain > 0 and not (self.maximum_match(pre_idx, cap_remain) == 0 and len(self.open_rings) == self.MAX_NESTED_BONDS):
                    candidates.append(1)
            
            rule = self.rand_rule(node, candidates)
            yield

            if rule == 1: # RB -> ringbond RB                
                assert cap_remain > 0
                r = self.get_node(node, 'RB', 1)
                r.task = node.task
                cap_remain = yield from self.tree_generator(r, cap_remain = cap_remain, is_last=is_last)
                node.add_child(r)
        elif node.symbol == 'BB':
            b = self.get_node(node, 'branch', 0)
            candidates = [0]
            assert cap_remain > 0
            if cap_remain > 1:
                candidates.append(1)
            rule = self.rand_rule(node, candidates)
            yield

            if rule == 1: # BB -> branch BB
                rest = yield from self.tree_generator(b, cap_remain=cap_remain - 1, is_last=False)
                node.add_child(b)
                bb = self.get_node(node, 'BB', 1)
                rest = yield from self.tree_generator(bb, cap_remain=rest + 1, is_last=is_last)
                node.add_child(bb)
            else:
                rest = yield from self.tree_generator(b, cap_remain=cap_remain, is_last=is_last)
                node.add_child(b)

            cap_remain = rest
        elif node.symbol == 'ringbond':            
            pre_idx = node.get_pre()
            mm = self.maximum_match(pre_idx, cap_remain)
            if node.task:
                assert mm > 0 and mm >= len(self.open_rings)
            
            candidates = []
            # whether to match bond
            if mm > 0 and len(self.open_rings):
                for r in self.open_rings:
                    if self.ring_valid(r, pre_idx, cap_remain):
                        candidates.append(r)
            # whether to create bond
            if mm == 0 or (not node.task and len(self.open_rings) < self.MAX_NESTED_BONDS):
                assert len(self.open_rings) < self.MAX_NESTED_BONDS
                candidates.append(self.MAX_NESTED_BONDS)
            
            r = self.rand_att(node, candidates)
            yield

            bond_idx = r
            bond_type = '?'
            create = False
            if r == self.MAX_NESTED_BONDS: # create new bond
                for i in range(self.MAX_NESTED_BONDS):
                    if not i in self.open_rings and ((not i in self.sameatom_bonds[self.atom_num - 1]) or cmd_args.bondcompact):
                        bond_idx = i
                        create = True
                        break
                assert create
            else: # paired bond removed
                assert r in self.open_rings
                self.matched_bonds.add((self.open_rings[r].pos, self.atom_num - 1))
                bond_type = self.open_rings[r].b_type
                del self.open_rings[r]

            self.sameatom_bonds[self.atom_num - 1].add(bond_idx)
            if bond_idx + 1 <= 9:
                d = self.get_node(node, 'DIGIT', -1)
                r = self.get_node(d, '\'%d\'' %  (bond_idx + 1), 0)

                d.add_child(r)
                node.add_child(d)
                if not create and bond_type is not None:
                    rule = self.rand_rule(node, [1])
                else:
                    rule = self.rand_rule(node, [0, 1])
            else:
                e = self.get_node(node, '\'%\'', -3)
                node.add_child(e)

                d1 = self.get_node(node, 'DIGIT', -2)
                r1 = self.get_node(d1, '\'%d\'' % ((bond_idx + 1) // 10), 0)

                d1.add_child(r1)
                node.add_child(d1)

                d2 = self.get_node(node, 'DIGIT', -1)
                r2 = self.get_node(d2, '\'%d\'' %  ((bond_idx + 1) % 10), 0)

                d2.add_child(r2)
                node.add_child(d2)
                if not create and bond_type is not None:
                    rule = self.rand_rule(node, [3])
                else:
                    rule = self.rand_rule(node, [2, 3])
            yield

            if rule % 2 == 1: # ringbond -> bond DIGIT | bond '%' DIGIT DIGIT
                b = self.get_node(node, 'bond', 0)
                if create:
                    yield from self.tree_generator(b, cap_remain=cap_remain, is_last=is_last)
                    bond_type = b.children[0].symbol
                else:
                    assert cap_remain >= self.bond_valence[bond_type]
                    b.allowed = [0, 1, 2, 3, 4]
                    yield from self.tree_generator(b, cap_remain=cap_remain, is_last=is_last)
                cap_remain -= self.bond_valence[b.children[0].symbol]
                node.add_child(b, 0)
            else:
                if bond_type == '?':
                    bond_type = None
                cap_remain -= 1

            if create:
                assert bond_type is None or bond_type != '?'
                self.open_rings[bond_idx] = RingBond(self.atom_num - 1, bond_type)
        elif node.symbol == 'branch':
            node.add_child(self.get_node(node, '\'(\'', 0))
            c = self.get_node(node, 'chain', -2)
            yield from self.tree_generator(c, left_conn=True, right_conn=False, is_last=is_last)
            rule = self.rand_rule(node)
            yield
            cost = 1

            if rule == 1: # branch -> '(' bond chain ')'
                b = self.get_node(node, 'bond', 1)
                yield from self.tree_generator(b, cap_remain= min(cap_remain, c.left_remain + 1), is_last=is_last)
                cost = self.bond_valence[b.children[0].symbol]
                node.add_child(b)
            node.add_child(c)
            node.add_child(self.get_node(node, '\')\'', -1))
            cap_remain -= cost
        elif node.symbol == 'BAI':
            rule = self.rand_rule(node)
            yield
            if rule % 2 == 0: # BAI -> isotope xxx
                i = self.get_node(node, 'isotope', 0)
                yield from self.tree_generator(i, is_last=is_last)
                node.add_child(i)

            s = self.get_node(node, 'symbol', -1 - (rule < 2))
            s.banned_set = set(['\'B\''])
            yield from self.tree_generator(s, left_conn=left_conn, right_conn=right_conn, is_last=is_last)
            node.atom_pos = s.atom_pos
            node.add_child(s)

            cap = s.left_remain
            if rule <= 1: # BAI -> isotope aliphatic_organic BAC | aliphatic_organic BAC
                b = self.get_node(node, 'BAC', -1)
                cap = yield from self.tree_generator(b, cap_remain=cap, ref_symbol=s.children[0].symbol, is_last=is_last)
                node.add_child(b)            
            node.left_remain = cap
            node.right_remain = cap
            node.single_atom = True
        elif node.symbol == 'BAC':
            rule = self.rand_rule(node)
            yield
            if rule == 0 or rule == 2: # BAC -> chiral BAH | chiral
                c = self.get_node(node, 'chiral', 0)
                yield from self.tree_generator(c, is_last=is_last)
                node.add_child(c)
            if rule <= 1: # BAC -> chiral BAH | BAH
                b = self.get_node(node, 'BAH', -1)
                cap_remain = yield from self.tree_generator(b, cap_remain=cap_remain, ref_symbol=ref_symbol, is_last=is_last)
                node.add_child(b)
        elif node.symbol == 'BAH':
            if cap_remain == 0:
                rule = self.rand_rule(node, [0, 1])
            else:
                rule = self.rand_rule(node)
            yield
            if rule <= 1: # BAH -> hcount charge | charge
                c = self.get_node(node, 'charge', -1)
                borrow = 0
                if cap_remain > 0 and rule == 0:
                    borrow = 1
                cap_remain = yield from self.tree_generator(c, cap_remain=cap_remain - borrow, ref_symbol=ref_symbol, is_last=is_last)
                cap_remain += borrow                
                node.add_child(c)
            if rule % 2 == 0: # BAH -> hcount charge | hcount
                assert cap_remain > 0
                hc = self.get_node(node, 'hcount', 0)
                cap_remain = yield from self.tree_generator(hc, cap_remain=cap_remain, is_last=is_last)
                node.add_child(hc, 0)
        elif node.symbol == 'hcount':
            rule = self.rand_rule(node)
            yield
            h = self.get_node(node, '\'H\'', 0)
            node.add_child(h)

            cost = 1
            if rule == 1: # hcount -> 'H' DIGIT
                d = self.get_node(node, 'DIGIT', -1)
                yield from self.tree_generator(d, cap_remain=cap_remain, is_last=is_last)
                cost = int(d.children[0].symbol[1 : -1])
                node.add_child(d)
            cap_remain -= cost            
        elif node.symbol == 'charge':            
            if cap_remain == 0:
                rule = self.rand_rule(node, [2, 3])
            else:
                rule = self.rand_rule(node)
            yield

            if rule <= 1: # charge -> '-' | '-' DIGIT
                m = self.get_node(node, '\'-\'', 0)
                node.add_child(m)
                cost = 1
                if rule == 1: # charge -> '-' DIGIT
                    d = self.get_node(node, 'DIGIT', -1)
                    yield from self.tree_generator(d, cap_remain=cap_remain, is_last=is_last)
                    cost = int(d.children[0].symbol[1 : -1])
                    node.add_child(d)
                cap_remain -= cost
            else: # charge -> '+' | '+' DIGIT
                p = self.get_node(node, '\'+\'', 0)
                node.add_child(p)                
                delta = 1
                if rule == 1: # charge -> '+' DIGIT
                    d1 = self.get_node(node, 'DIGIT', -1)
                    yield from self.tree_generator(d1, is_last=is_last)
                    delta = int(d1.children[0].symbol[1 : -1])
                    node.add_child(d1)
                cap_remain += delta            
            assert ref_symbol is not None and ref_symbol != '\'B\''
        elif node.symbol == 'DIGIT':
            if cap_remain is None or cap_remain > len(self.prod[node.symbol]):
                rule = self.rand_rule(node)
            else:
                rule = self.rand_rule(node, range(cap_remain))
            yield

            d = self.get_node(node, '\'%d\'' % (rule + 1), 0)
            node.add_child(d)
        else:
            assert node.symbol in ['smiles', 'atom', 'bracket_atom', 'isotope', 'chiral', 'symbol']
            rule = self.rand_rule(node)
            yield
            p = self.prod[node.symbol][rule]

            for i in range(len(p)):
                c = self.get_node(node, p[i], i)
                if not p[i][0] == '\'': # non-terminal
                    t = yield from self.tree_generator(c, left_conn, right_conn, cap_remain=cap_remain, ref_symbol=ref_symbol, is_last=is_last)
                    node.left_remain = c.left_remain
                    node.right_remain = c.right_remain
                    node.single_atom = c.single_atom                    
                    node.atom_pos = c.atom_pos
                    node.is_aromatic = c.is_aromatic
                    if t >= 0:
                        cap_remain = t
                node.add_child(c)                

        if cap_remain is not None:
            assert cap_remain >= 0
            return cap_remain
        return -1

    def decode(self, node, walker):
        self.walker = walker
        self.walker.reset()
        self.reset_state()
        self.generator = self.tree_generator(node, is_last=True)


def create_tree_decoder(utils):
    fname = cmd_args.grammar_file.split('/')[-1]
    # print('using', fname)
    tree_decoder = AttMolGraphDecoder(utils)
    return tree_decoder

if __name__ == '__main__':
    pass
