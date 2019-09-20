# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:15:05 2018

@author: vikuv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
import math

N=10000
d=10
total_reward = 0
ads_selected = []
no_of_selections = [0] * d
sums_of_rewards = [0] * d

for n in range(0,N):
    ad = 0 
    max_upper_bound = 0
    for i in range(0,d):
        if (no_of_selections[i] > 0):
            average_reward = sums_of_rewards[i]/no_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/no_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    no_of_selections[ad] = no_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
