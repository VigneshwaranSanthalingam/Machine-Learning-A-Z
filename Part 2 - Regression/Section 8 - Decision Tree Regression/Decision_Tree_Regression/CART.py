# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:27:46 2018

@author: vikuv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.cross_validation import train_test_split
XT,Xt,yT,yt = train_test_split(X, y, test_size = 0.2, random_state = 0 )

from sklearn.tree import DecisionTreeRegressor
mdl = DecisionTreeRegressor()
mdl.fit(X,y)
pred = mdl.predict(6.5)

from sklearn.ensemble import RandomForestRegressor
mdl1 = RandomForestRegressor(n_estimators = 3000, random_state = 0)
mdl1.fit(X,y)
pred1 = mdl1.predict(6.5)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, mdl1.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()