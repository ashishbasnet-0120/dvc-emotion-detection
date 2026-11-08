import numpy as np
import pandas as pd
import os  # for creating files and folder 
import yaml
from sklearn.model_selection import train_test_split

import logging

logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')


console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Define log format
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Attach formatter to handler
console_handler.setFormatter(formatter)

# Attach handler to logger
logger.addHandler(console_handler)


logger.info("Data ingestion started")
testing_size=yaml.safe_load(open('params.yaml','r'))['data_ingestion']['test_size']
df=pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

df.drop(columns=['tweet_id'],inplace=True)

final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()

final_df['sentiment'] = final_df['sentiment'].map({
    'happiness': 1,
    'sadness': 0
})



train_data,test_data=train_test_split(final_df,test_size=testing_size,random_state=42)

data_path=os.path.join("data","processed")  # os.path.join() combines multiple folder and file names into a single valid path

#"raw" represents the main folder.



#"data" represents a subfolder inside raw.



#os.path.join() combines them into one path.


os.makedirs(data_path,exist_ok=True)

train_data.to_csv(os.path.join(data_path,'train.csv'))
test_data.to_csv(os.path.join(data_path,'test.csv'))

logger.info("Data ingestion completed successfully")