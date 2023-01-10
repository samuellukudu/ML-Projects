import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

Config = {
    "num_splits": 5
}

clfs = []

imp = np.zeros(len(features))
for clf in clfs[:Config['num_splits']]:
    imp += clf.feature_importances_
    
# print('------------------------ XGBOOST --------------------------------')
plt.barh([features[i] for i in np.argsort(imp/Config['num_splits'])], sorted(imp/Config['num_splits']))
plt.title('Xgboost feature importance')
plt.xlabel('Feature importance')
plt.show()


imp = np.zeros(len(features))
for clf in clfs[Config['num_splits']:Config['num_splits']+5]:
    imp += clf.feature_importances_
    
# print('------------------------ LGBM --------------------------------')
plt.barh([features[i] for i in np.argsort(imp/Config['num_splits'])], sorted(imp/Config['num_splits']))
plt.title('LGBM feature importance')
plt.xlabel('Feature importance')
plt.show()


imp = np.zeros(len(features))
for clf in clfs[Config['num_splits']+5:]:
    imp += clf.feature_importances_
    
# print('------------------------ LGBM --------------------------------')
plt.barh([features[i] for i in np.argsort(imp/Config['num_splits'])], sorted(imp/Config['num_splits']))
plt.title('Catboost feature importance')
plt.xlabel('Feature importance')
plt.show()