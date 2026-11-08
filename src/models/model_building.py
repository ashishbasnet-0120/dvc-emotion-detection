import numpy as np
import pandas as pd 
import os
import pickle
import yaml
from sklearn.ensemble import GradientBoostingClassifier



params=yaml.safe_load(open('params.yaml','r'))['model_building']
#fetch the data
test_data=pd.read_csv('./data/external/feature_test.csv')
train_data=pd.read_csv('./data/external/feature_train.csv')


X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values

# Define and train the XGBoost model
clf = GradientBoostingClassifier(n_estimators=params['n_estimators'],learning_rate=params['learning_rate'])

clf.fit(X_train, y_train)


# save 

pickle.dump(clf,open('model.pkl','wb'))