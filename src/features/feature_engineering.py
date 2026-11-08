import numpy as np
import pandas as pd
import os 
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
features=yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']

#fetch the data
test_data=pd.read_csv('./data/interim/preprocessed_test.csv')
train_data=pd.read_csv('./data/interim/preprocessed_train.csv')




# Handle missing values
train_data['content'] = train_data['content'].fillna('')
test_data['content'] = test_data['content'].fillna('')

# Apply Bag of Words
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values



# Handle missing values
train_data['content'] = train_data['content'].fillna('')
test_data['content'] = test_data['content'].fillna('')


# Apply Bag of Words (CountVectorizer)
vectorizer = TfidfVectorizer(max_features=features)

# Fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)

train_df = pd.DataFrame(X_train_bow.toarray())

train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())

test_df['label'] = y_test



data_path=os.path.join("data","external")
os.makedirs(data_path,exist_ok=True)

train_df.to_csv(os.path.join(data_path,'feature_train_tfidf.csv'))
test_df.to_csv(os.path.join(data_path,'feature_test_tfidf.csv'))