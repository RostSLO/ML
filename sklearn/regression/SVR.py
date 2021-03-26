'''
Created on Feb 08, 2021
@author: rboruk
'''

from sklearn import svm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np

##Support Vector Machines - https://scikit-learn.org/stable/modules/svm.html#classification
#Support Vector Regression

#read data
boston = load_boston()

#create pandas dataframe
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)

print(boston_df.head())

np.random.seed(31)

#create the data
X = boston_df.drop('target', axis=1)
y = boston_df['target']

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.2)

#instantiate and fit model
regr = svm.SVR()
regr.fit(X_train, y_train)

#evaluate the scores
resScore = regr.score(X_test, y_test)
print(f'Score function result: {resScore:.2f}%')

#Evaluating model using cross validation
resCrossVal = cross_val_score(regr, X, y, cv=5)
print(f'Cross Validation score function result: {resCrossVal}')
cross_val_mean = np.mean(resCrossVal)
print(f'Cross validation mean result: {cross_val_mean:.2f}%')