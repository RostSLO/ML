'''
Created on Feb 07, 2021
@author: rboruk
'''

#Support Vector Machines - https://scikit-learn.org/stable/modules/svm.html#classification
#Support Vector Classification


from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#read data from the file
heart_disease = pd.read_csv('heart-disease.csv')

np.random.seed(31)

#create the data
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']

#split into triaining and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Instantiate and fit model
clf = svm.SVC()
clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)


#Evaluate the model using score()
resScore = clf.score(X_test, y_test)
print(f'Score function result: {resScore:.2f}%')

#Evaluating model using cross validation
resCrossVal = cross_val_score(clf, X, y, cv=5)
print(f'Cross Validation score function result: {resCrossVal}')
cross_val_mean = np.mean(resCrossVal)
print(f'Cross validation mean result: {cross_val_mean:.2f}%')

#review confusion map


#set the font scale
sns.set(font_scale=1.5)

#create a confusion matrix
conf_mat = confusion_matrix(y_test, y_preds)

#plot a confusion matrix using Seaborn heatmap
tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()
matrix = np.array([[tp, fp], [fn, tn]])

# plot
sns.heatmap(matrix, annot=True, cmap="viridis", fmt='g')
plt.xticks([0.5, 1.5], labels=[1, 0])
plt.yticks([0.5, 1.5], labels=[1, 0])
plt.title('Confusion matrix')
plt.xlabel('Actual label')
plt.ylabel('Predicted label')
plt.show()

print(classification_report(y_test, y_preds))