import argparse
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from joblib import Parallel, delayed

def singleSimilarity(line, inf2, tanimoto=False):
    for line2 in inf2:
        if tanimoto:
            m_x = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(line.strip()), 2)
            m_y = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(line2.strip()), 2)
            s = DataStructs.TanimotoSimilarity(m_x, m_y)
        else:
            m_x = FingerprintMols.FingerprintMol(Chem.MolFromSmiles(line.strip()))
            m_y = FingerprintMols.FingerprintMol(Chem.MolFromSmiles(line2.strip()))
            s = DataStructs.FingerprintSimilarity(m_x, m_y)
        if s > 0.5:
            return None
    return line.strip()


def smilaritywithzinc(infile, n_threads=1, tanimoto=False):
    inf2 = [_.strip() for _ in open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      'zinc_small.txt'), 'r').readlines()]
    num_lines = sum(1 for _ in open(infile, 'r').readlines()) - 1
    with open(infile, 'r') as inf1:
        next(inf1)
        smile_list = Parallel(n_jobs=n_threads)\
            (delayed(singleSimilarity)(_, inf2, tanimoto) for _ in
             tqdm(inf1, desc="Similarity to ZINC:", total=num_lines))
    smile_list = list(set(filter(None.__ne__, smile_list)))
    print(len(smile_list))
    return smile_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='similarity with zinc')
    parser.add_argument('--infile', '-i', type=str, help='input file')
    parser.add_argument('--outfile', '-o', type=str, default='out_similarity.txt', help='output file')
    parser.add_argument('--n_workers', '-n', type=int, default=1, help="Number of workers")
    parser.add_argument('--tanimoto', '-t', action="store_true", help="Use Tanimoto Similarity")

    args = parser.parse_args()
    if args.infile and args.outfile:
        smilelist = smilaritywithzinc(args.infile, args.n_workers, args.tanimoto)
        df = pd.DataFrame({'SMILES': smilelist})
        df.to_csv(args.outfile, index=False)
