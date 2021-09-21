import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import operator
from rdkit.Chem.Fingerprints import FingerprintMols


def clusterfps(fps, cutoff=0.2):
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])

    # now cluster the data:
    cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return cs


########################## get the top 5 dissimilar with centroid mols in cluster
# cluster_new = tuple()

def topmol(fps):
    topmols = []
    less5mol = []
    centroid_list = []
    clusters = clusterfps(fps, cutoff=0.4)
    print(len(clusters))
    n = 0
    for i in range(len(clusters)):
        centroid = clusters[i][0]
        centroid_list.append(centroid)
        if len(clusters[i]) > 5:
            n += 1
            index = list(clusters[i])
            score = {}
            for j in range(len(index) - 1):
                index = list(clusters[i])
                index.pop(j)
                sims = DataStructs.BulkTanimotoSimilarity(fps[index[j]], [fps[x] for x in index])
                score[index[j]] = sum(sims)
            score_sort = sorted(score.items(), key=operator.itemgetter(1))
            # dists.extend([1-x for x in sims])
            df_mol = pd.DataFrame(score_sort, columns=['index', 'score'])
            top5mols = list(df_mol['index'][0:5])
            # cluster_new = cluster_new + (clusters[i], )
            topmols = topmols + top5mols
        else:
            less5mol += clusters[i]
    print(n)
    return topmols, less5mol, centroid_list



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='clustering')
    parser.add_argument('--infile', '-i', type=str, help='input file')
    parser.add_argument('--outfile', '-o', type=str, default='out_cluster.txt', help='output file')

    args = parser.parse_args()

    if args.infile and args.outfile:
        ms = [x for x in Chem.SmilesMolSupplier(args.infile, nameColumn=-1)]
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in ms]
        molindex, less5mol, centroids = topmol(fingerprints)
        print('5 in cluster')
        print(len(molindex))
        print('less 5 mol')
        print(len(less5mol))
        print('centroids')
        print(len(centroids))
        allindex = molindex + less5mol
        smilelist = [Chem.MolToSmiles(ms[x]) for x in centroids]
        df = pd.DataFrame({'SMILES': smilelist})
        df.to_csv(args.outfile, index=False)
    else:
        print("input file and output file is empty!")
