#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from dglt.contrib.moses.moses.metrics.utils import get_n_rings, get_mol

#_mcf = pd.read_excel("MCF_all.xlsx", names=['smarts'])
#_filters = list(set([Chem.MolFromSmarts(x) for x in _mcf['smarts'].values]))

# filters = [x for x in _filters if x]



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

allowed_2atoms = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Cl', 'Ar',
                  'Ca', 'Ti', 'Cr', 'Fe', 'Ni', 'Cu', 'Ga', 'Ge', 'As', 'Se',
                  'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
                  'Pd', 'Ag', 'Cd', 'Sb', 'Te', 'Xe', 'Ba', 'La', 'Ce', 'Pr',
                  'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Er', 'Tm', 'Yb',
                  'Lu', 'Hf', 'Ta', 'Re', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
                  'Bi', 'At', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Pu', 'Am', 'Cm',
                  'Bk', 'Cf', 'Es', 'Fm', 'Md', 'Lr', 'Rf', 'Db', 'Sg', 'Mt',
                  'Ds', 'Rg', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
allowed_1atoms = ['F', 'N', 'n', 'c', 'C', 'O', 'o', 'S', 's', 'H']

allowed = allowed_1atoms + allowed_2atoms
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
    # ring_info = mol.GetRingInfo()
    # if ring_info.NumRings() != 0 and any(
    #         len(x) >= 8 for x in ring_info.AtomRings()
    # ):
    #     return False
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
        return False
    # if any(atom.GetSymbol() in not_allowed for atom in mol.GetAtoms()):
    #     return False
    # # limitation of atoms number
    # atom_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
    # if atom_list.count('Cl') > 3:
    #     return False
    # if atom_list.count('Br') > 1:
    #     return False
    # if atom_list.count('F') > 5:
    #     return False
    # # MCF filter
    # h_mol = Chem.AddHs(mol)
    # if any(h_mol.HasSubstructMatch(smarts) for smarts in filters):
    #     return False
    # patt = Chem.MolFromSmiles('N(=O)[O-]')
    # if len(mol.GetSubstructMatches(patt, uniquify=True)) >= 2:
    #     return False
    return True


def process_filter(infile):
    mols = Chem.SmilesMolSupplier(infile, nameColumn=0)
    # mols = Chem.SDMolSupplier(infile)
    mol_list = []
    for i, m in enumerate(mols):
        if m is None:
            continue
        if filter_atom(m):
            smiles = Chem.MolToSmiles(m, isomericSmiles=False)
            mol_list.append(smiles)
    return mol_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='filter moleculars with properties')
    parser.add_argument('--infile', '-i', type=str, help='input file')
    parser.add_argument('--outfile', '-o', type=str, default='out_mcf.txt', help='output file')

    args = parser.parse_args()

    if args.infile and args.outfile:
        mollist = process_filter(args.infile)
        df = pd.DataFrame({'SMILES': mollist})
        df.to_csv(args.outfile, index=False)
    else:
        print("input file and output file is empty!")

