'''
Created on March 07, 2021
@author: rboruk
'''

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.dates as dts

import csv
import datetime as dt

#preparing pandas dataframe
data = pd.read_csv("california.csv", sep=",")

#pre-processing to get only required information
data = data[["date_time", "WindGustKmph", "humidity", "precipMM", "pressure", "tempC", "winddirDegree", "windspeedKmph"]]
#print(data.head())
#print(str(len(data)))

#set proper datatime index and keep only day time weather
dataIndex = pd.DatetimeIndex(data['date_time'].astype(str))
data.index = dataIndex

#filter the data
data['date_time'] = pd.to_datetime(data['date_time'])
data = data[data['date_time'].dt.hour == 12]

data['days'] = data['date_time'].dt.day
#data[data['hours'] != 0]

#transform datetime to ordinal
data['date_time'] = data["date_time"].dt.strftime('%Y-%m')
#data['date_time'] = data['date_time'].map(dt.datetime.toordinal)

#shuffle data
data = shuffle(data, random_state=42)

predict = "tempC"
#split the data trainig elements and results
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

#X = data.drop([predict], axis=1)
#Y = data[predict]

#split the target data to the training and testing sets of data - 80% trainint, 20% testing
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

#create linear regression object
linear = linear_model.LinearRegression()

#train the model on train data
linear.fit(X_train, y_train)
#find the accuracy
acc = linear.score(X_test, y_test)

#make prediction of the weathre on test data
predictC = linear.predict(X_test)

print(X_test.head())
dates = dts.date2num(X_test[:, 0])
print(dates[0])

# Plot outputs
#plot the actual weather
plt.plot_date(dates, y_test,  color='green', label="Actual data", solid_joinstyle='miter')
#plot the predicted weather
plt.plot_date(dates, predictC, color='blue', label="Predicted data")

#titles for the labels and the plot
plt.xlabel('Sample')
plt.ylabel('Predicted Temperature')
plt.title(f"Real weather in California versus predicted by ML with accuracy = {str(round(acc*100, 2))}%")
#show the legend
plt.legend()
#display plot
plt.show()
