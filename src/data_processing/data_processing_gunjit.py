import pandas as pd

data = pd.read_csv('../data/interim/ODI_all.csv')

data['Strike Rate'] = data.apply(lambda x: (x['Runs'] / x['Balls Faced']) * 100 if x['Balls Faced'] >= 20 else -1, axis=1)
data['Economy Rate'] = data.apply(lambda x: (x['Runsgiven'] / x['Balls Bowled']) * 6 if x['Balls Bowled'] > 0 else -1, axis=1)
data['Duck Out'] = data.apply(lambda x: 1 if x['Runs'] == 0 and x['Outs']==1 else 0, axis=1)
data.to_csv('../data/interim/ODI_gunjit.csv', index=False)