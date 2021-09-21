import pandas as pd
import argparse

parser = argparse.ArgumentParser('Generate fake data')
parser.add_argument('--origin',
                    requires=True, type=str,
                    help='The original file')
parser.add_argument('--recons',
                    requires=True, type=str,
                    help='The reconstructed file')
parser.add_argument('--output',
                    type=str, default=None,
                    help='The output filename')
parser.add_argument('--uniq',
                    action='store_true',
                    help='Remove duplicated smiles')

opts = parser.parse_args()
orign_data = pd.read_csv(opts.origin)
data = pd.read_csv(opts.recons)
if opts.output is None:
    opts.output = opts.origin.strip('.csv') + '_fake.csv'

out_lst = []
for or_row, row in zip(orign_data.values, data.values):
    for smiles in row:
        if not isinstance(smiles, float):
            out_lst.append([smiles, *(or_row[1:].flatten())])

out_data = pd.DataFrame(out_lst, columns=orign_data.columns)
if opts.uniq:
    out_data.drop_duplicates(['SMILES'], inplace=True)

out_data.to_csv(opts.output, index=False)