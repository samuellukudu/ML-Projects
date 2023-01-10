# src/cross_validation
# Ilustration using California Housing dataset
# Part of Kaggle Tabular competition Episode 1, Season 3

from sklearn.model_selection import KFold

def create_folds(df):
    X = df.drop('MedHouseVal', axis=1)
    X = X.sample(frac=1.0).reset_index(drop=True)
    y = df.loc[:, 'MedHouseVal']
    kfold = KFold(n_splits=Config['num_splits'], shuffle=True, random_state=Config['seed'])
    # tscv = model_selection.TimeSeriesSplit(n_splits=Config['num_splits'], test_size=Config['test_size'], gap=Config['gap'])
    df['kfold'] = -1

    for fold, (tr_idx, val_idx) in enumerate(tqdm(kfold.split(X), total=Config['num_splits'])):
        df.loc[val_idx, 'kfold'] = fold
        
    return df