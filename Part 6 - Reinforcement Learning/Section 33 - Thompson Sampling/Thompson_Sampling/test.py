# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 21:43:24 2018

@author: vikuv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
import random

N=10000
d=10
total_reward = 0
ads_selected = []
no_of_reward_1 = [0] * d
no_of_reward_0 = [0] * d

for n in range(0,N):
    ad = 0 
    max_random = 0
    for i in range(0,d):
        random_beta = random.betavariate(no_of_reward_1[i] + 1, no_of_reward_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        no_of_reward_1[ad] = no_of_reward_1[ad] + 1
    else:
        no_of_reward_0[ad] = no_of_reward_0[ad] + 1
    total_reward = total_reward + reward
    
plt.hist(ads_selected)
plt.title("Thompson_sampling")
plt.xlabel('Ads')
plt.ylabel('No. of times selected')
