'''
Created on February 14, 2021

@author rboruk
'''

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_boston
from sklearn import tree

from dtreeviz.trees import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # load the two data sets
    iris = load_iris()
    boston = load_boston()

    # prepare the data
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit the classifier
    clf = tree.DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    #tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True, filled=True)

    viz = dtreeviz(clf,
                    x_data = X_train,
                    y_data = y_train,
                    target_name = 'class',
                    feature_names = iris.feature_names,
                    class_names = list(iris.target_names),
                    title = "Decision Tree - Iris data set")
    viz