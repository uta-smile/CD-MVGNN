# Changelog 
* all featurizer related import was modified to  `import from moses.moses.data.featurizer`
* some utils function is imported from deepchem package
    * atomic_coordinates.py:16  `from deepchem.utils import rdkit_util, pad_array1`
    * atomic_coordinates.py:17  `from deepchem.utils.rdkit_util import MoleculeLoadException`
    * binding_pocket_features.py:12  `from deepchem.utils.save import log`
    * coulomb_matrices.py:16 `from deepchem.utils import pad_array`
    * graph_features.py:11 `from deepchem.data.datasets import DiskDataset`
    * nn_score_utils.py:15  `import deepchem.utils.rdkit_util as rdkit_util`
    * rdkit_grid_featurizer.py:22 `from deepchem.utils.save import log`
    * rdkit_grid_featurizer.py:23 `from deepchem.utils.rdkit_util import load_molecule`
    * rdkit_grid_featurizer.py:24 `from deepchem.utils.rdkit_util import MoleculeLoadException`