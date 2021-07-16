import pandas as pd

df = pd.read_csv('processed.cleveland.data')
df = df.dropna()
df['num'] = (df['num']>0)+0
print(df['num'].unique())
df.to_csv('./heartDisease.csv',index=False)
