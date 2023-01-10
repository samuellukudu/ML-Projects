# src/LGBM_model.py
# The California Housing Dataset is used for illustration

import lightgbm as lgbm
from lightgbm.sklearn import LGBMRegressor

clfs = []
err = []
def lgbm_model(df, fold):
    df_train = df[df['kfold'] != fold]
    df_valid = df[df['kfold'] == fold]
    
    df_train.drop('kfold', axis=1, inplace=True)
    df_valid.drop('kfold', axis=1, inplace=True)
    
    X_train = df_train.drop('MedHouseVal', axis=1)
    y_train = df_train['MedHouseVal']
    
    X_valid = df_valid.drop('MedHouseVal', axis=1)
    y_valid = df_valid['MedHouseVal']
    
    clf = lgbm.LGBMRegressor(learning_rate=0.01,
                             max_depth=9,
                             num_leaves=90,
                             colsample_bytree=0.8,
                             subsample=0.9,
                             subsample_freq=5,
                             min_child_samples=36,
                             reg_lambda=28,
                             n_estimators=20000,
                             metric='rmse',
                             random_state=42)
    
    clf.fit(X_train.values, y_train,
            eval_set=[(X_valid.values, y_valid)], 
            callbacks=[lgbm.early_stopping(100, verbose=True)])
    
    preds = clf.predict(X_valid)
    
    rmse = mean_squared_error(y_valid, preds, squared=False)
    err.append(rmse)
    clfs.append(clf)
    print(f'RMSE on fold {i}: {rmse}')
    print('-'*50)


