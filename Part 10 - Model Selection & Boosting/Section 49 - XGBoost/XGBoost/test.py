# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

if __name__ == '__main__':
        
    from xgboost import XGBClassifier as xgbc
    mdl = xgbc()
    mdl.fit(X_train, y_train)
    # Part 3 - Making the predictions and evaluating the model
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = mdl, X = X_train, y = y_train, cv = 10 )
    acc = accuracies.mean()
    std = accuracies.std()
    from sklearn.model_selection import GridSearchCV
    parameters = [{'max_depth' : np.linspace( 1, 10, num = 11).astype(int), 'learning_rate' : np.linspace(0,1,num = 20) }
                    ]
    grid = GridSearchCV(estimator = mdl, param_grid = parameters, scoring = 'accuracy', cv = 10)
    grid = grid.fit(X_train, y_train)
    best_acc = grid.best_score_
    best_params = grid.best_params_