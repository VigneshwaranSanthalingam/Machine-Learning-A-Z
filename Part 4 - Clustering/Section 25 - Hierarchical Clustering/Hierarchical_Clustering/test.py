# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 20:48:19 2018

@author: vikuv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Points in dataset')
plt.ylabel('Euclidian distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering as ac
mdl = ac(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
ypred = mdl.fit_predict(X)

plt.scatter(X[ypred == 0, 0], X[ypred == 0, 1], c = 'red', label = 'careful')
plt.scatter(X[ypred == 1, 0], X[ypred == 1, 1], c = 'blue', label = 'standard')
plt.scatter(X[ypred == 2, 0], X[ypred == 2, 1], c = 'violet', label = 'target')
plt.scatter(X[ypred == 3, 0], X[ypred == 3, 1], c = 'orange', label = 'careless')
plt.scatter(X[ypred == 4, 0], X[ypred == 4, 1], c = 'green', label = 'sensible')
plt.title('Hierarchial Clustering')
plt.xlabel('Salary')
plt.ylabel('Spending Score')
plt.legend()
plt.show()