import torch
import operator
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

def got_all_Type_solu_dic(dataset_file, bondtype_freq=20, atomtype_freq=10, column_name='smiles'):
    df = pd.read_csv(dataset_file)
    column_name = column_name
    sample_elems = df[column_name].tolist()

    bondtype_dic = {}
    atomtype_dic = {}
    for smile in sample_elems:  # Wierd, the len(data) is longer, but no data was in the rest of part.
        if len(smile) == 0:
            continue
        try:
            mol = MolFromSmiles(smile)
            bondtype_dic = fillBondType_dic(mol, bondtype_dic)
            atomtype_dic = fillAtomType_dic(mol, atomtype_dic)
        except AttributeError:
            pass
        else:
            pass

    sorted_bondtype_dic = sorted(bondtype_dic.items(), key=operator.itemgetter(1))
    sorted_bondtype_dic.reverse()
    bondtype_list_order = [ele[0] for ele in sorted_bondtype_dic]
    bondtype_list_number = [ele[1] for ele in sorted_bondtype_dic]

    filted_bondtype_list_order = []
    for i in range(0, len(bondtype_list_order)):
        if bondtype_list_number[i] > bondtype_freq:
            filted_bondtype_list_order.append(bondtype_list_order[i])
    filted_bondtype_list_order.append('Others')

    sorted_atom_types_dic = sorted(atomtype_dic.items(), key=operator.itemgetter(1))
    sorted_atom_types_dic.reverse()
    atomtype_list_order = [ele[0] for ele in sorted_atom_types_dic]
    atomtype_list_number = [ele[1] for ele in sorted_atom_types_dic]

    filted_atomtype_list_order = []
    for i in range(0, len(atomtype_list_order)):
        if atomtype_list_number[i] > atomtype_freq:  # filtered low freq atom, threshold 10 ???
            filted_atomtype_list_order.append(atomtype_list_order[i])
    filted_atomtype_list_order.append('Others')
    return filted_bondtype_list_order, filted_atomtype_list_order

# bond type dict {bond_type: number}
def fillBondType_dic(rdmol, bondtype_dic):
    # Add bonds
    for bond in rdmol.GetBonds():
        BeginAtom, EndAtom = bond.GetBeginAtom(), bond.GetEndAtom()
        begin_idx, end_idx = BeginAtom.GetAtomicNum(), EndAtom.GetAtomicNum()
        if begin_idx < end_idx:
            bond_type = str(begin_idx) + '_' + str(end_idx)
        else:
            bond_type = str(end_idx) + '_' + str(begin_idx)
        if bond_type in bondtype_dic.keys():
            bondtype_dic[bond_type] += 1
        else:
            bondtype_dic[bond_type] = 1
    return (bondtype_dic)

# atom type dict {atom_type: number}
def fillAtomType_dic(rdmol, atomtype_dic):
    for atom in rdmol.GetAtoms():
        atom_num = atom.GetAtomicNum()
        if atom_num in atomtype_dic:
            atomtype_dic[atom_num] += 1
        else:
            atomtype_dic[atom_num] = 1
    return (atomtype_dic)

def mol_collate_func(batch):
    adj_list = []
    afm_list = []
    size_list = []
    bft_list = []
    orderAtt_list, aromAtt_list, conjAtt_list, ringAtt_list = [], [], [], []

    for datum in batch:
        size_list.append(datum[0].shape[0])
    max_size = np.max(size_list)  # max of batch    222 for hiv, 132 for tox21,
    btf_len = datum[2].shape[0]
    # max_size = max_molsize #max_molsize 132
    # padding
    for datum in batch:
        filled_adj = np.zeros((max_size, max_size), dtype=np.float32)
        filled_adj[0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[0]
        filled_afm = np.zeros((max_size, 25), dtype=np.float32)
        filled_afm[0:datum[0].shape[0], :] = datum[1]
        filled_bft = np.zeros((btf_len, max_size, max_size), dtype=np.float32)
        filled_bft[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[2]

        filled_orderAtt = np.zeros((5, max_size, max_size), dtype=np.float32)
        filled_orderAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[3]

        filled_aromAtt = np.zeros((3, max_size, max_size), dtype=np.float32)
        filled_aromAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[4]

        filled_conjAtt = np.zeros((3, max_size, max_size), dtype=np.float32)
        filled_conjAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[5]

        filled_ringAtt = np.zeros((3, max_size, max_size), dtype=np.float32)
        filled_ringAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[6]

        adj_list.append(filled_adj)
        afm_list.append(filled_afm)
        bft_list.append(filled_bft)
        orderAtt_list.append(filled_orderAtt)
        aromAtt_list.append(filled_aromAtt)
        conjAtt_list.append(filled_conjAtt)
        ringAtt_list.append(filled_ringAtt)

    if torch.cuda.is_available():
        return ([torch.from_numpy(np.array(adj_list)).cuda(), torch.from_numpy(np.array(afm_list)).cuda(),
                 torch.from_numpy(np.array(bft_list)).cuda(), torch.from_numpy(np.array(orderAtt_list)).cuda(),
                 torch.from_numpy(np.array(aromAtt_list)).cuda(), torch.from_numpy(np.array(conjAtt_list)).cuda(),
                 torch.from_numpy(np.array(ringAtt_list)).cuda()])
    else:
        return ([torch.from_numpy(np.array(adj_list)), torch.from_numpy(np.array(afm_list)),
                 torch.from_numpy(np.array(bft_list)), torch.from_numpy(np.array(orderAtt_list)),
                 torch.from_numpy(np.array(aromAtt_list)), torch.from_numpy(np.array(conjAtt_list)),
                 torch.from_numpy(np.array(ringAtt_list))])

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    if classname.find('Conv2d') != -1:
        m.weight.data.fill_(1.0)

def feature_normalize(x_all):
    """Min Max Feature Scalling for Atom Feature Matrix"""
    feature_num = x_all[0][0][1].shape[1]
    feature_min_dic = {}
    feature_max_dic = {}
    for i in range(len(x_all)):
        afm = x_all[i][0][1]
        afm_min = afm.min(0)
        afm_max = afm.max(0)
        for j in range(feature_num):
            if j not in feature_max_dic.keys():
                feature_max_dic[j] = afm_max[j]
                feature_min_dic[j] = afm_min[j]
            else:
                if feature_max_dic[j] < afm_max[j]:
                    feature_max_dic[j] = afm_max[j]
                if feature_min_dic[j] > afm_min[j]:
                    feature_min_dic[j] = afm_min[j]

    for i in range(len(x_all)):
        afm = x_all[i][0][1]
        feature_diff_dic = {}
        for j in range(feature_num):
            feature_diff_dic[j] = feature_max_dic[j] - feature_min_dic[j]
            if feature_diff_dic[j] == 0:
                feature_diff_dic[j] = 1
            afm[:, j] = (afm[:, j] - feature_min_dic[j]) / (feature_diff_dic[j])
        x_all[i][0][1] = afm
    return x_all

class AdditionalInfo:
    def __init__(self, dataset_file, smiles_field="mol"):
        self.bond_type_dict, self.atom_type_dict = got_all_Type_solu_dic(dataset_file, column_name=smiles_field)
        self.dict_len = len(self.bond_type_dict) + len(self.atom_type_dict)
