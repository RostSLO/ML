'''
Created on March 13, 2021
@author: rboruk
'''

# Regular EDA (exploratory data analysis) and plotting libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML models from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve

df = pd.read_csv('heart-disease.csv')

'''
# Data exploration EDA
df.shape
df['target'].value_counts()
df['target'].value_counts().plot(kind='bar', color=['salmon', 'lightblue']);

df.info()
df.isna().sum()

df.describe()

pd.crosstab(df.target, df.sex)

pd.crosstab(df.target, df.sex).plot(kind='bar',
                                    figsize=(10, 6),
                                    color = ['salmon', 'lightblue'])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('0 = No Disease, 1 = Disease')
plt.ylabel('Amount')
plt.legend(['Female', 'Male'])
plt.xticks(rotation=0)

# Create another figure
plt.figure(figsize=(10, 6))

# Scatter with postivie examples
plt.scatter(df.age[df.target==1],
            df.thalach[df.target==1],
            c="salmon")

# Scatter with negative examples
plt.scatter(df.age[df.target==0],
            df.thalach[df.target==0],
            c="lightblue")

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);

# Check the distribution of the age column with a histogram
df.age.plot.hist();

# Make a correlation matrix
corr_matrix = df.corr()

fig, ax = plt.subplots(figsize=(15, 10))

ax = sns.heatmap(corr_matrix,
                annot=True,
                linewidth=0.5,
                fmt='.2f',
                cmap='YlGnBu')

plt.show();
'''

# MODELING

X = df.drop('target', axis=1)
y = df['target']

# Split data to train and test datasets
np.random.seed(31)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

# Put models in a dictionary
models = {'Logistic Regression': LogisticRegression(),
          'KNN': KNeighborsClassifier(),
          'Random Forest': RandomForestClassifier()}

# Function to fit and score our models
def fit_and_score(models, X_train, X_test, y_train, y_test):

    # Set random seed
    np.random.seed(31)
    # Make a dictionary to keep our model scores
    model_scores = {}
    for name, model in models.items():
        # fit te model from dictionary
        model.fit(X_train, y_train)
        # Evaluate the model and save its score in model_scores dict
        model_scores[name] = model.score(X_test, y_test)

    return model_scores


model_scores = fit_and_score(models,
                             X_train,
                             X_test,
                             y_train,
                             y_test)

print(model_scores)

model_compare = pd.DataFrame(model_scores, index=['accuracy'])
model_compare.T.plot.bar()
plt.show();




