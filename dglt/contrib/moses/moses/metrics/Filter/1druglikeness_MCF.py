#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from dglt.contrib.moses.moses.metrics.utils import get_n_rings, get_mol
from tqdm import tqdm
from joblib import Parallel, delayed

_mcf = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), "MCF_all.xlsx"),
                     names=['smarts'])
# _mcf = pd.read_csv("MCF_new_.csv", names=['smarts'])
_filters = list(set([Chem.MolFromSmarts(x) for x in _mcf['smarts'].values]))
# _filters = []
# n = 0
# for x in _mcf['smarts'].values:
#     smart = Chem.MolFromSmarts(x)
#     if smart is None:
#         n += 1
#         print(x)
#     elif smart in _filters:
#         continue
#     else:
#         _filters.append(smart)
# print(n)
# remove None in _filters
filters = [x for x in _filters if x]

print(len(filters))
# _pains = pd.read_csv('wehi_pains.csv', names=['smarts', 'names'])
# _filters = [Chem.MolFromSmarts(x) for x in _mcf.append(_pains, sort=True)['smarts'].values]


# smiles5 = ['Cc1ccc2c(Nc3cccc(C(F)(F)F)c3)noc2c1C#Cc1cnc2cccnn12',
#            'O=C(Nc1cccc(C(=O)N2CCN(c3ccnc4[nH]ccc34)C2)c1)c1cccc(C(F)(F)F)c1',
#            'Cc1ccc(CNC(=O)Cc2cccc(F)c2)cc1-c1ccc2[nH]ncc2c1',
#            'Cc1cc2[nH]c(-c3cc(CN(C)C)cc(C(F)(F)F)c3)nc2cc1C#Cc1cncnc1',
#            'Cc1ccc(NC(=O)CN2CCC2C(F)(F)F)cc1NC(=O)c1cnn2c1CCC2']
#
# for smile in smiles5:
#     mol = Chem.MolFromSmiles(smile)
#     h_mol = Chem.AddHs(mol)
#     for smart in filters:
#         if h_mol.HasSubstructMatch(smart):
#             print('The pair is')
#             print(Chem.MolToSmarts(smart))
#             print(Chem.MolToSmiles(mol))

AcceptorSmarts = [
    '[oH0;X2]',
    '[OH1;X2;v2]',
    '[OH0;X2;v2]',
    '[OH0;X1;v2]',
    '[O-;X1]',
    '[SH0;X2;v2]',
    '[SH0;X1;v2]',
    '[S-;X1]',
    '[nH0;X2]',
    '[NH0;X1;v3]',
    '[$([N;+0;X3;v3]);!$(N[C,S]=O)]'
]
Acceptors = []
for hba in AcceptorSmarts:
    Acceptors.append(Chem.MolFromSmarts(hba))

# allowed_2atoms = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Cl', 'Ar',
#                   'Ca', 'Ti', 'Cr', 'Fe', 'Ni', 'Cu', 'Ga', 'Ge', 'As', 'Se',
#                   'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
#                   'Pd', 'Ag', 'Cd', 'Sb', 'Te', 'Xe', 'Ba', 'La', 'Ce', 'Pr',
#                   'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Er', 'Tm', 'Yb',
#                   'Lu', 'Hf', 'Ta', 'Re', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
#                   'Bi', 'At', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Pu', 'Am', 'Cm',
#                   'Bk', 'Cf', 'Es', 'Fm', 'Md', 'Lr', 'Rf', 'Db', 'Sg', 'Mt',
#                   'Ds', 'Rg', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
# allowed_1atoms = ['F', 'N', 'n', 'c', 'C', 'O', 'o', 'S', 's', 'H']
#
# allowed = allowed_1atoms + allowed_2atoms

not_allowed = ['Si', 'Co', 'P', 'Hg', 'I', 'Fe', 'As', 'Sb', 'Zn', 'Se', 'se', 'Te', 'B', 'Si']



def properties_filter(mol):
    """
    Calculates the properties that contain logP, MW, HBA, HBD, TPSA, NRB
    """
    #frag = Chem.rdmolops.GetMolFrags(mol)  # remove '.'
    #if len(frag) > 1:
        #return False

    MW_s = Descriptors.MolWt(mol)  # MW
    if MW_s < 250 or MW_s > 750:
        return False

    ALOGP_s = Descriptors.MolLogP(mol)  # ALOGP
    if ALOGP_s < -2 or ALOGP_s > 7:
        return False

    HBA_s = 0
    for hba in Acceptors:  # HBA
        if mol.HasSubstructMatch(hba):
            matches = mol.GetSubstructMatches(hba)
            HBA_s += len(matches)

    HBD_s = Descriptors.NumHDonors(mol)  # HBD

    if HBA_s + HBD_s >= 10:
        return False

    TPSA_s = Descriptors.TPSA(mol)  # TPSA
    if TPSA_s >= 150:
        return False

    NRB_s = Descriptors.NumRotatableBonds(mol)  # NRB
    if NRB_s >= 10:
        return False

    return True


def filter_atom(mol):
    """
        Checks if mol
        * has more than 7 rings
        * has only allowed atoms
        * is not charged
    """
    mol = get_mol(mol)
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(
            len(x) >= 8 for x in ring_info.AtomRings()
    ):
        return False
    # limitation of aromatic ring number
    # if Chem.rdMolDescriptors.CalcNumAromaticRings(mol) > 4:
    #     return False
    # if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
    #     return False
    # if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
    #     return False
    if any(atom.GetSymbol() in not_allowed for atom in mol.GetAtoms()):
        return False
    # limitation of atoms number
    atom_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if atom_list.count('Cl') > 3:
        return False
    if atom_list.count('Br') > 1:
        return False
    if atom_list.count('F') > 5:
        return False
    # MCF filter
    h_mol = Chem.AddHs(mol)
    if any(h_mol.HasSubstructMatch(smarts) for smarts in filters):
        return False
    # if any(mol.HasSubstructMatch(smarts) for smarts in MCF_mol):
    #     return False
    patt = Chem.MolFromSmiles('N(=O)[O-]')
    if len(mol.GetSubstructMatches(patt, uniquify=True)) >= 2:
        return False
    return True


def single_filter(m):
    if m is None:
        return None
    if filter_atom(m) and properties_filter(m):
        smiles = Chem.MolToSmiles(m, isomericSmiles=False)
        return smiles
    return None


def process_filter1(infile, n_threads=1):
    mols = Chem.SmilesMolSupplier(infile, nameColumn=0)
    num_smiles = sum(1 for _ in open(infile, 'r').readlines()) - 1
    # mols = Chem.SDMolSupplier(infile)
    mol_list = Parallel(n_jobs=n_threads)\
        (delayed(single_filter)(_)
         for _ in tqdm(mols, desc="Drug likeness MCF:", total=num_smiles))
    mol_list = list(filter(None.__ne__, mol_list))
    return mol_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='filter moleculars with properties')
    parser.add_argument('--infile', '-i', type=str, help='input file')
    parser.add_argument('--outfile', '-o', type=str, default='out_mcf.txt', help='output file')
    parser.add_argument('--n_workers', '-n', type=int, default=1, help="Number of workers")

    args = parser.parse_args()

    if args.infile and args.outfile:
        mollist = process_filter1(args.infile, args.n_workers)
        df = pd.DataFrame({'SMILES': mollist})
        df.to_csv(args.outfile, index=False)
    else:
        print("input file and output file is empty!")
