'''
Created on March 20, 2021
@author: rboruk
'''

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

# read test and validation data
df = pd.read_csv('TrainAndValid.csv')

# learn data EDA (exploratory data analysis)
df