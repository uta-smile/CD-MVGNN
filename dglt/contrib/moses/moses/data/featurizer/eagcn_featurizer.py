from rdkit import Chem
import numpy as np
from dglt.contrib.moses.moses.data.featurizer import Featurizer
from dglt.contrib.moses.moses.data.featurizer import neural_fp
from deepchem.data.datasets import DiskDataset


class EagcnFeaturizer(Featurizer):
    name = 'eagcn'

    def __init__(self, bond_type_dict, atom_type_dict):
        self.bond_type_dict = bond_type_dict
        self.atom_type_dict = atom_type_dict

    def _featurize(self, mol):
        try:
            graph = neural_fp.molToGraph(mol, self.bond_type_dict, self.atom_type_dict)
            graph_feature = graph.dump_as_matrices_Att()  # todo: feature not normalized
            return graph_feature
        except AttributeError:
            print('Mol {} has an error'.format(Chem.MolToSmiles(mol)))
        except TypeError:
            print('Mol {} can not convert to graph structure'.format(Chem.MolToSmiles(mol)))
        except ValueError:
            return np.array([])