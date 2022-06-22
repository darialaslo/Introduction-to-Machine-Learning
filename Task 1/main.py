import scipy as sp
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

data_train = pd.read_csv("./train.csv")

label = data_train.iloc[:,0]
features = data_train.iloc[:,1:]

vect_lambdas = [0.1, 1, 10, 100, 200]
vect_rmse = []
for num_lambda in vect_lambdas:
    list_rmse = []
    kfold = KFold(n_splits=10)
    for train_index, test_index in kfold.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = label[train_index], label[test_index]
        ridge = Ridge(alpha=num_lambda)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        list_rmse.append(mean_squared_error(y_test, y_pred, squared=False))
    vect_rmse.append(np.mean(list_rmse))

submission = pd.DataFrame(vect_rmse)
submission.to_csv('./submission.csv', index=False, header=False)