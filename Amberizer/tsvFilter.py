import pandas as pd

df = pd.read_csv('../sabdab_summary_all.tsv', sep="\t", header=0, dtype=str)

df1 = df.loc[:, ('pdb', 'Hchain', 'Lchain', 'antigen_chain', 'antigen_type', 'resolution', 'r_free', 'r_factor')]
df1.dropna(how='any', inplace=True)
df1.drop_duplicates(subset=['pdb'], keep='first', inplace=True)

df1[['pdb', 'Hchain', 'Lchain', 'antigen_chain', 'antigen_type']] = df1[
    ['pdb', 'Hchain', 'Lchain', 'antigen_chain', 'antigen_type']].astype(str)
df1.replace('unknown', pd.NA, inplace=True)
df1.dropna(inplace=True)
df1[['resolution', 'r_free', 'r_factor']] = df1[['resolution', 'r_free', 'r_factor']].astype(float)

# filtering
df1 = df1[~df1['pdb'].str.contains('\+', regex=True)]
df1 = df1[df1['antigen_type'].str.contains('protein|nucleic', regex=True)]
filterQuery = f"resolution <= 2.5"
df1 = df1.copy().query(filterQuery)
filterQuery = f"r_factor <= 0.25"
df1 = df1.copy().query(filterQuery)

# Save csv
df1.to_csv('ab_ag_DB_filtered_protNUCL.csv', sep=',', index=False)
