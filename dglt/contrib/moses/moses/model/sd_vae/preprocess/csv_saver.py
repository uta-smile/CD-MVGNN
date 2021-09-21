import numpy as np

from dglt.contrib.moses.moses.model.sd_vae.config import get_parser

cmd_args, _ = get_parser().parse_known_args()

def csv_writer(one_hot, masks, out_file, smiles_list=None):
    with open(out_file, 'w') as f:
        for i, (a, b) in enumerate(zip(one_hot, masks)):
            a_inds = "|".join([str(_) for _  in a])
            b_inds1 = "|".join(str(_) for _ in np.where(b)[0])
            b_inds2 = "|".join(str(_) for _ in np.where(b)[1])
            string = " ".join([a_inds, b_inds1, b_inds2])
            if smiles_list is not None:
                f.write(",".join([smiles_list[i], string]) + "\n")
            else:
                f.write(string + "\n")

def csv_reader(file_name, utils):
    true_binary = []
    rule_masks = []
    for line in open(file_name, 'r'):
        a_inds, b_inds1, b_inds2 = line.strip().split(',')[-1].split()

        a = np.zeros([cmd_args.max_decode_steps, utils.DECISION_DIM])
        a_inds = [int(_) for _ in a_inds.split("|")]
        num_steps = len(a_inds)
        a[np.arange(num_steps), a_inds] = 1
        a[np.arange(num_steps, cmd_args.max_decode_steps), -1] = 1
        true_binary.append(a)
        b = np.zeros([cmd_args.max_decode_steps, utils.DECISION_DIM])
        b_inds1 = [int(_) for _ in b_inds1.split("|")]
        b_inds2 = [int(_) for _ in b_inds2.split("|")]
        b[b_inds1, b_inds2] = 1
        b[np.arange(num_steps, cmd_args.max_decode_steps), -1] = 1
        rule_masks.append(b)
    return np.stack(true_binary), np.stack(rule_masks)

def line_reader(line, utils, onehot=True):
    a_inds, b_inds1, b_inds2 = line.strip().split()

    if onehot:
        a = np.zeros([cmd_args.max_decode_steps, utils.DECISION_DIM])
        a_inds = [int(_) for _ in a_inds.split("|")]
        num_steps = len(a_inds)
        a[np.arange(num_steps), a_inds] = 1
        a[np.arange(num_steps, cmd_args.max_decode_steps), -1] = 1
    else:
        a = np.array([int(_) for _ in a_inds.split("|")])
        num_steps = len(a)

    b = np.zeros([cmd_args.max_decode_steps, utils.DECISION_DIM])
    b_inds1 = [int(_) for _ in b_inds1.split("|")]
    b_inds2 = [int(_) for _ in b_inds2.split("|")]
    b[b_inds1, b_inds2] = 1
    b[np.arange(num_steps, cmd_args.max_decode_steps), -1] = 1

    return (a, b)
