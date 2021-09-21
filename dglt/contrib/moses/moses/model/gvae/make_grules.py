"""Parse raw-formatted SMILES grammar into readable format."""

import os
import shutil

def expand_qst_n_ast_symbols(rhs_meta):
    """Expand question & asterisk symbols."""

    def __append_rhs_to_list(words, rhs, idx, rhs_list):
        if idx == len(words):
            rhs_list.append(rhs.strip())
            return
        word = words[idx]
        if word.endswith('?') or word.endswith('*'):
            word = word[:-1]
            __append_rhs_to_list(words, rhs, idx + 1, rhs_list)
        __append_rhs_to_list(words, rhs + ' ' + word, idx + 1, rhs_list)

    words = rhs_meta.split()
    words_w_ast = set([word[:-1] for word in words if word.endswith('*')])
    rhs_list = []
    __append_rhs_to_list(words, '', 0, rhs_list)

    return rhs_list, words_w_ast

def expand_elp_symbol(rhs_prev, rhs_next):
    """Expand the ellipis symbol."""

    def __split_word(word):
        for idx in range(len(word) - 2, -1, -1):
            if not word[idx].isdigit():
                break
        return word[1:idx + 1], int(word[idx + 1:-1])

    sfx_prev, int_prev = __split_word(rhs_prev)
    sfx_next, int_next = __split_word(rhs_next)
    assert sfx_prev == sfx_next
    rhs_list = ['\'%s%d\'' % (sfx_prev, x) for x in range(int_prev + 1, int_next)]

    return rhs_list

def parse_line(line_str):
    """Parse a line of raw-formatted SMILES grammar."""

    sub_strs = [sub_str.strip() for sub_str in line_str.split('::=')]
    assert len(sub_strs) == 2
    lhs = sub_strs[0]
    rhs_list = [sub_str.strip() for sub_str in sub_strs[1].split('|')]
    rules = []

    words_w_ast = set()
    for idx, rhs in enumerate(rhs_list):
        if '?' in rhs or '*' in rhs:
            rhs_list_tmp, words_w_ast_tmp = expand_qst_n_ast_symbols(rhs)
            words_w_ast |= words_w_ast_tmp
        elif rhs == '...':
            assert (idx - 1 >= 0) and (idx + 1 < len(rhs_list))
            rhs_list_tmp = expand_elp_symbol(rhs_list[idx - 1], rhs_list[idx + 1])
        else:
            rhs_list_tmp = [rhs]
        for rhs_tmp in rhs_list_tmp:
            rules.append('%s -> %s' % (lhs, rhs_tmp))

    return rules, words_w_ast

def merge_rules(rules_full):
    """Merge rules with same LHS."""

    rules_merged = []
    lhs, rhs_list = None, []
    for idx, rule in enumerate(rules_full):
        lhs_curr, rhs_curr = [sub_str.strip() for sub_str in rule.split('->')]
        if idx == 0:
            lhs, rhs_list = lhs_curr, [rhs_curr]
        elif lhs == lhs_curr:
            rhs_list.append(rhs_curr)
        else:
            rules_merged.append('%s -> %s' % (lhs, ' | '.join(rhs_list)))
            lhs, rhs_list = lhs_curr, [rhs_curr]
    if lhs:
        rules_merged.append('%s -> %s' % (lhs, ' | '.join(rhs_list)))

    return rules_merged

### Main Entry ###

root_dir = '/data1/jonathan/Molecule.Generation/AIPharmacist'
grm_path_raw = os.path.join(root_dir, 'moses/model/gvae/smiles.grammar.raw')
grm_path_fll = os.path.join(root_dir, 'moses/model/gvae/smiles.grammar.full')
grm_path_mrg = os.path.join(root_dir, 'moses/model/gvae/smiles.grammar.merged')
rls_path = os.path.join(root_dir, 'moses/model/gvae/grules.txt')

# parse the raw SMILSE grammar
with open(grm_path_raw, 'r') as i_file:
    rules = []
    words_w_ast = set()
    for i_line in i_file:
        rules_tmp, words_w_ast_tmp = parse_line(i_line.strip())
        rules += rules_tmp
        words_w_ast |= words_w_ast_tmp
    if words_w_ast:
        nb_rules = len(rules)
        for idx in range(nb_rules):
            lhs, rhs = [sub_str.strip() for sub_str in rules[idx].split('->')]
            if lhs in words_w_ast:
                rules.append('%s -> %s %s' % (lhs, lhs, rhs))
    rules.sort()
    for idx in range(len(rules)):
        if rules[idx].startswith('smiles'):
            rules = [rules[idx]] + rules[:idx] + rules[idx + 1:]
            break

# write the full SMILES grammar to file
with open(grm_path_fll, 'w') as o_file:
    o_file.write('\n'.join(rules))
shutil.copyfile(grm_path_fll, rls_path)

# write the merged SMILES grammar to file
rules_merged = merge_rules(rules)
with open(grm_path_mrg, 'w') as o_file:
    o_file.write('\n'.join(rules_merged))
