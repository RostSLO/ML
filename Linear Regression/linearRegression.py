'''
Created on March 05, 2021
@author: rboruk
'''

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import csv

#preparing pandas dataframe
data = pd.read_csv("student-mat.csv", sep=";")

#shuffle data
data = shuffle(data, random_state=22)

#pre-processing to get only required information
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

print(data.head())

predict = "G3"
#split the data trainig elements and results
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

#X = data.drop([predict], axis=1)
#Y = data[predict]

#split the target data to the training and testing sets of data
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

#create linear regression object
linear = linear_model.LinearRegression()

#train the model on train data
linear.fit(X_train, y_train)
#find the accuracy
acc = linear.score(X_test, y_test)

#make prediction of the marks on test data
predictG3 = linear.predict(X_test)

#create list of # of samples used for prediction
# will be used to plot the actual and predicted data by samples
axisX = [i for i in range(len(predictG3))]

# Plot outputs
#plot the actual marks
plt.plot(axisX, y_test,  color='green', label="Actual data")
#plot the predicted marks
plt.plot(axisX, predictG3, color='blue', label="Predicted data")

#titles for the labels and the plot
plt.xlabel('Sample')
plt.ylabel('G3 - mark for the year')
plt.title(f"Real marks versus predicted by ML with accuracy = {str(round(acc*100, 2))}%")
#show the legend
plt.legend()
#display plot
plt.show()
