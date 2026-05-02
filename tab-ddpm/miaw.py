import pandas as pd
df = pd.read_csv('data/churn/churn_raw.csv')
df = df.sample(frac=1, random_state=0).reset_index(drop=True)
n = int(len(df) * 0.8)
df.iloc[:n].to_csv('data/churn/train.csv', index=False)
df.iloc[n:].to_csv('data/churn/test.csv', index=False)
print('wrote data/churn/train.csv and data/churn/test.csv')