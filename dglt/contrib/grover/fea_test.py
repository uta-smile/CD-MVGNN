# from dglt.data.featurization import get_available_features_generators
# from dglt.data.featurization import get_available_features_generators
from collections import Counter

from rdkit import Chem

from dglt.contrib.grover.mol2features import register_features_generator
from dglt.contrib.grover.vocab import AtomVocab
from dglt.data.featurization import get_features_generator

register_features_generator('fgtasklabel')

features_generator = get_features_generator('fgtasklabel')
# generator = rdDescriptors.RDKit2D(RDKIT_PROPS)
smiles = "CC(C)Nc1c(nc2ncccn12)c3ccc4[nH]ncc4c3"
mol = Chem.MolFromSmiles(smiles)
atoms = mol.GetAtoms()
print(atoms)
for i, atom in enumerate(mol.GetAtoms()):
    nei = Counter()
    nei_bond = Counter
    for a in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx())
        # nei[str(a.GetAtomicNum())+str(bond.GetBondType())] +=1
        nei[str(a.GetSymbol()) + "-" + str(bond.GetBondType())] += 1
        # print(str(a.GetAtomicNum()))
        # print(str(bond.GetBondType()))
        # nei_bond[int(bond.GetBondType())] +=1
        # nei.append(a.GetAtomicNum())

        # nei_bond.append(bond.GetBondType())
    print(nei)
    # print(nei_bond)

# features = rdkit_functional_group_label_features_generator(smiles)
# print(features)

vocab = AtomVocab.load_vocab("../drug_data/all_smi_tryout.pkl")
