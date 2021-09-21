import pandas as pd
from tqdm import tqdm
cls_dataset = ['pcba', 'muv', 'hiv', 'bace', 'bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'chembl']
df_merged = None
for dataset in tqdm(cls_dataset):
    dataset = dataset + '.csv'
    df = pd.read_csv(dataset)
    df = df.groupby(df['smiles']).sum(min_count=1).reset_index()
    if df_merged is None:
        df_merged = df
    else:
        df_merged = pd.merge(df_merged, df, how='outer', on='smiles')

df_merged.to_csv('merged.csv', index=False)
