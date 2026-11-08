import numpy as np
import pandas as pd
import os

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer


nltk.download('wordnet')
nltk.download('stopwords')


test_data=pd.read_csv('./data/processed/test.csv')
train_data=pd.read_csv('./data/processed/train.csv')

# 1. Lowercase conversion
def lower_case(text):
    if not isinstance(text, str):
        return ""
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

# 2. Stop words removal
def remove_stop_words(text):
    if not isinstance(text, str):
        return ""
    stop_words = set(stopwords.words("english"))
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

# 3. Number removal
def removing_numbers(text):
    if not isinstance(text, str):
        return ""
    text = ''.join([i for i in text if not i.isdigit()])
    return text

# 4. Punctuation and extra whitespace removal
def removing_punctuations(text):
    if not isinstance(text, str):
        return ""
    # Remove standard punctuation marks
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = text.replace(':', '')
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

# 5. URL removal
def removing_urls(text):
    if not isinstance(text, str):
        return ""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# 6. Lemmatization
def lemmatization(text):
    if not isinstance(text, str):
        return ""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

# 7. Remove rows with fewer than 3 words in a DataFrame
def remove_small_sentences(df):
    for i in range(len(df)):
        if len(str(df.text.iloc[i]).split()) < 3:
            df.text.iloc[i] = np.nan
    return df

# 8. Full pipeline for a Pandas DataFrame column ('content')
def normalize_text(df):
    df.content = df.content.apply(lambda content: lower_case(content))
    df.content = df.content.apply(lambda content: remove_stop_words(content))
    df.content = df.content.apply(lambda content: removing_numbers(content))
    df.content = df.content.apply(lambda content: removing_punctuations(content))
    df.content = df.content.apply(lambda content: removing_urls(content))
    df.content = df.content.apply(lambda content: lemmatization(content))
    return df


train_processed_data = normalize_text(train_data)
test_processed_data = normalize_text(test_data)


data_path=os.path.join("data","interim")
os.makedirs(data_path,exist_ok=True)

train_processed_data.to_csv(os.path.join(data_path,'preprocessed_train.csv'))
test_processed_data.to_csv(os.path.join(data_path,'preprocessed_test.csv'))