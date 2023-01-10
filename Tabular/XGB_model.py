# src/XGB_model.py
# baseline xgb regressor model

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

clfs = []
err = []
def xgb_model(df, fold):
    df_train = df[df['kfold'] != fold]
    df_valid = df[df['kfold'] == fold]
    
    df_train.drop('kfold', axis=1, inplace=True)
    df_valid.drop('kfold', axis=1, inplace=True)
    
    X_train = df_train.drop('MedHouseVal', axis=1)
    y_train = df_train['MedHouseVal']
    
    X_valid = df_valid.drop('MedHouseVal', axis=1)
    y_valid = df_valid['MedHouseVal']
    
    clf = XGBRegressor(n_estimators=20000,
                       max_depth=9,
                       learning_rate=0.01,
                       colsample_bytree=0.66,
                       subsample=0.9,
                       min_child_weight=22,
                       reg_lambda=16,
                       early_stopping_rounds=100, 
                       tree_method='hist', # gpu_hist
                       seed=42)
    
    clf.fit(X_train.values, y_train,
            eval_set=[(X_valid.values, y_valid)], 
            verbose=1000)
    
    preds = clf.predict(X_valid)
    
    rmse = mean_squared_error(y_valid, preds, squared=False)
    err.append(rmse)
    clfs.append(clf)
    print(f'RMSE on fold {i}: {rmse}')
    print('-'*50)