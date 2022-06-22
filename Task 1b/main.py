

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data_train = pd.read_csv("./train.csv")
feature_names = []
for i in range(21):
    name = "x" + str(i+1)
    feature_names.append(name)

y = data_train.iloc[:,1]
linear = data_train.iloc[:,2:7]
quadratic = np.power(linear,2)
exponential = np.exp(linear)
cosine = np.cos(linear)
constant = pd.DataFrame(np.ones(len(linear)))
result = pd.concat([linear, quadratic, exponential, cosine, constant], join="inner", axis=1)
result.columns = feature_names

linreg = LinearRegression().fit(result, y)
submission = pd.DataFrame(linreg.coef_)
submission.to_csv('./submission.csv', index=False, header=False)