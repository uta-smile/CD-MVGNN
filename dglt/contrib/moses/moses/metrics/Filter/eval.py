import argparse
from rdkit import Chem


smiles5 = ['Cc1ccc2c(Nc3cccc(C(F)(F)F)c3)noc2c1C#Cc1cnc2cccnn12',
           'O=C(Nc1cccc(C(=O)N2CCN(c3ccnc4[nH]ccc34)C2)c1)c1cccc(C(F)(F)F)c1',
           'Cc1ccc(CNC(=O)Cc2cccc(F)c2)cc1-c1ccc2[nH]ncc2c1',
           'Cc1cc2[nH]c(-c3cc(CN(C)C)cc(C(F)(F)F)c3)nc2cc1C#Cc1cncnc1',
           'Cc1ccc(NC(=O)CN2CCC2C(F)(F)F)cc1NC(=O)c1cnn2c1CCC2']

def eval(infile):
    mols = Chem.SmilesMolSupplier(infile, nameColumn=0)
    smiles = [Chem.MolToSmiles(x) for x in mols]
    for i in range(len(smiles5)):
        if smiles5[i] in smiles:
            print('True')
        else:
            print('False')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='whether import molecular is included')
    parser.add_argument('--infile', '-i', type=str, help='input file')

    args = parser.parse_args()
    eval(args.infile)
