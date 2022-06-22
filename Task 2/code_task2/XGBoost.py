import pandas as pd
import numpy as np
from xgboost import XGBClassifier
#from sklearn import svm

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (GridSearchCV, cross_val_score, KFold)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from sklearn.svm import SVC

train_features = pd.read_csv("./data/train_features.csv")
train_features = train_features.sort_values(by=['pid', 'Time'])

test_features = pd.read_csv("./data/test_features.csv")
test_features = test_features.sort_values(by=['pid', 'Time'])

train_labels = pd.read_csv("./data/train_labels.csv")
train_labels = train_labels.sort_values(by='pid')

#Pre-processing of training and test data
def calculate_features(data, timepoints):
    x=[]
    features = [np.nanmedian, np.nanmean, np.nanvar, np.nanmin, np.nanmax]
    no_patients=int(data.shape[0]/timepoints)
    median=np.nanmedian(data, axis=0)
    
    for i in range(no_patients):
        #making sure the first and last entry of the selection is from the same patient 
        assert data[timepoints*i, 0] == data[timepoints * (i+1) -1, 0]
        
        #selecting data 
        tests_patient= data[timepoints*i: timepoints*(i+1), 2:]
        
        #if all entries in the column are NaNs replace with median
        sum_na=np.count_nonzero(np.isnan(tests_patient), axis=0)

        for i in range (2, data.shape[1]):
            if sum_na[i-2]==timepoints:
                tests_patient[:,i-2]=np.nan_to_num(tests_patient[:,i-2],nan=median[i])
        
        #create empty array for the new features
        new_features=np.empty((len(features),data[:, 2:].shape[1]))
        
        #calculating the features
        for i, feature in enumerate(features):
            new_features[i] = feature(tests_patient, axis=0)
        x.append(new_features.ravel())
    return np.array(x)


x_train = calculate_features(train_features.to_numpy(), 12)
x_test = calculate_features(test_features.to_numpy(), 12)

# Sub-tasks
labels_1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 
            'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
y_train_1 = train_labels[labels_1].to_numpy()

y_train_2 = train_labels['LABEL_Sepsis'].to_numpy().ravel()

labels_3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
y_train_3 = train_labels[labels_3].to_numpy()

df = pd.DataFrame({'pid': test_features.iloc[0::12, 0].values})



##Task 1
print("Task 1")
#clf_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), SVC(gamma=2, C=1))
scaler = StandardScaler()
x_train_scale = scaler.fit_transform(x_train)
x_test_scale = scaler.fit_transform(x_test)

model= XGBClassifier()

for i, label in enumerate(labels_1):
    model.fit(x_train_scale, y_train_1[:, i].ravel())
    predictions = model.predict_proba(x_test_scale)[:, 1]
    df[label] = predictions
    scores = cross_val_score(model, x_train_scale, y_train_1[:, i], cv=5, scoring='roc_auc', verbose=True)
    print("Cross-validation score is {score:.3f}, standard deviation is {err:.3f}".format(score = scores.mean(), err = scores.std()))


##Task 2
print("Task 2")
model2=XGBClassifier()
model2.fit(x_train_scale, y_train_2)
predictions = model2.predict_proba(x_test_scale)[:, 1]
# print("Training score:", metrics.roc_auc_score(y_train_2, pipeline_t2.predict_proba(x_train)[:, 1]))
df['LABEL_Sepsis'] = predictions
scores = cross_val_score(model2, x_train_scale, y_train_2, cv=5, scoring='roc_auc', verbose=True)
print("Cross-validation score is {score:.3f}, standard deviation is {err:.3f}".format(score = scores.mean(), err = scores.std()))

##Task 3
print("Task 3")
reg_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), HistGradientBoostingRegressor(max_depth=3))
for i, label in enumerate(labels_3):
    pipeline_t3 = reg_pipeline.fit(x_train_scale, y_train_3[:, i])
    predictions = pipeline_t3.predict(x_test)
    print("Training score:", metrics.r2_score(y_train_3[:, i], pipeline_t3.predict(x_train_scale)))
    df[label] = predictions
    scores = cross_val_score(reg_pipeline, x_train_scale, y_train_3[:, i], cv=5, scoring='r2', verbose=True)
    print("Cross-validation score is {score:.3f}, standard deviation is {err:.3f}".format(score = scores.mean(), err = scores.std()))


# Submission
df.to_csv("./processed/submission_xgboost.zip", index=False, float_format='%.3f', compression='zip')