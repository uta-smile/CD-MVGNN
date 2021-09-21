from rdkit import Chem
import pandas as pd
import numpy as np
from third_party.dimorphite_dl.acid_base_pairs import get_pair, AcidBaseFinder

def mol_cls(smi_list):
    pair = get_pair()
    finder = AcidBaseFinder(pair)

    acid_list = []
    base_list = []

    for item in smi_list:
        mol = Chem.MolFromSmiles(item)
        smol = Chem.AddHs(mol)
        acid, base = finder.find_acid_base(smol)
        acid_list.append(acid)
        base_list.append(base)

    return np.column_stack((acid_list, base_list))


#if __name__ == '__main__':
#    mol_cls('/data2/weiyangxie/mit/drug_data/pka_classfication.csv')
    # class Sample:
    #     def __init__(self):
    #         self.smiles = ['ONC1C=CC=CC=1']
    # sample = Sample()
