'''
Created on March 20, 2021
@author: rboruk
'''


'''
!!!!!!!!!!!
Training, Validation and Testing data for this work can be downloaded here:
https://www.dropbox.com/s/hfivk0fmmwnfo8k/bluebook-for-bulldozers.zip?file_subpath=%2Fbluebook-for-bulldozers
!!!!!!!!!!!
'''


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt

# read test and validation data
df = pd.read_csv('../../../Data for ML/CutTrainAndValid.csv',
                 low_memory=False,
                 parse_dates=['saledate'])

# learn data EDA (exploratory data analysis)
print(df.dtypes)
# find missing data
print(df.isna().sum())

# plot hist diagram to see sales distribution over time
#fig, ax = plt.subplots(figsize=(10, 8))

#ax.scatter(df['saledate'], df['SalePrice'])
#plt.xlabel('Sale Date')
#plt.ylabel('Sale Price')
#plt.title('Sales Distribution over time')
#plt.show()

# plot bar graph to see sales distribution between states
#plt.bar(df['state'], df['SalePrice'])
#plt.xticks(rotation=90)
#plt.show()

# make a copy of data to modify
df_temp = df

# prepare data for training: make all column numeric and fix missing data

# add datetime parameters for sale data
df_temp['saleYear'] = df_temp.saledate.dt.year
df_temp['saleMonth'] = df_temp.saledate.dt.month
df_temp['saleDay'] = df_temp.saledate.dt.day
df_temp['dayOfWeek'] = df_temp.saledate.dt.dayofweek
df_temp['dayOfYear'] = df_temp.saledate.dt.dayofyear

# print(df_temp.head().T)

# remove saledate afer we enriched data with datetime columns
df_temp.drop('saledate', axis=1, inplace=True)

# check the values of different columns
print(df_temp.state.value_counts())

# check the dataframe information
print(df_temp.info())

# convert string to categories
for label, content in df_temp.items():
    if pd.api.types.is_string_dtype(content):
        df_temp[label] = content.astype('category').cat.as_ordered()

# fixing missing data for numeric columns
for label, content in df_temp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add a binary column which tells us if the data was missing or not
            df_temp[label + '_is_missing'] = pd.isnull(content)
            # fill missing numeric values with the median
            df_temp[label] = content.fillna(content.median())

# filling and tuning categorical variables into numbers

import secrets
secret_key = secrets.token_hex(16)
print(secret_key)
