import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier

#Import data
train_features = pd.read_csv("./data/train.csv")
test_features = pd.read_csv("./data/test.csv")

#Training set
features = []
for i in range(len(train_features)):
    features.append(list(train_features["Sequence"][i]))

features_pd = pd.DataFrame(features)
train = pd.concat([features_pd, train_features['Active']], axis=1)

#Test set 
tfeatures = []
for i in range(len(test_features)):
    tfeatures.append(list(test_features["Sequence"][i]))

test = pd.DataFrame(tfeatures)

#One-hot encoding category variables
categ = ['A', 'C', 'D', 'F', 'G', 'R', 'H', 'K', 'E','S', 'T', 'N', 'Q', 'U', 'P', 'I', 'L', 'M', 'W', 'Y', 'V']
categ.sort()
categories = [categ, categ,categ, categ]

#Transform categorical data using One-hot enconding
features = train.iloc[:, :4].astype(str)
features_test = test.astype(str)

enc = OneHotEncoder(sparse=False, categories=categories)
transformed_train = enc.fit_transform(features)
transformed_test = enc.fit_transform(features_test)

#Training labels
labels = np.array(train['Active'])

#Prediction
clf = MLPClassifier(random_state=1, early_stopping=True, validation_fraction=0.15).fit(transformed_train, labels)
pred = clf.predict(transformed_test)
submission = pd.DataFrame(pred)
submission.to_csv('./submission_15.csv', index=False, header=False)