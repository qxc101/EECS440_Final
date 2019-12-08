#!/usr/bin/env python
# coding: utf-8

# In[75]:


import os
import math
import operator
import numpy as np
import random
import pandas as pd
import GLAD
import glob, operator
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


music = pd.read_csv('music_mturk_answers.csv')
sentiment = pd.read_csv('sentiment_mturk_answers.csv')
rte = pd.read_csv('rte.anno.csv')
data1 = music.values
data2 = sentiment.values
data3 = rte.values


# In[6]:


#for sentiment data(data2)
workers = np.unique(data2[:,0])
samples = np.unique(data2[:,1])

labelM = np.full((workers.shape[0],samples.shape[0]),-1)
for i in range(data2.shape[0]):
    x = int(np.where(workers == data2[i,0])[0])
    y = int(np.where(samples == data2[i,1])[0])
    labelM[x][y] = toint(data2[i][5])


# In[7]:


def toint(char):
    if char == 'pos':
        return 1
    else:
        return 0


# In[31]:


def getP(alpha,beta):
    pM = np.empty([])
    for i in range(alpha.shape[0]):
        for j in range(beta.shape[0]):
            pM = np.append(pM, 1/(1+math.exp(-alpha[i]*beta[j])))
            print(i)
    pM = pM.reshape(alpha.shape[0],beta.shape[0])
    return pM


# In[74]:


fig = plt.figure()
ax = plt.axes()
ax.set(xlabel='number of labels', ylabel='accuracy',title='sentiment')
x = [5,10,20,40,80,150,200]
y = [0.777,0.795,0.810,0.807,0.815,0.825,0.832]
ax.plot(x, y,'-g', label ='sentiment',color='blue')


# In[62]:


np.unique(data3[:,1]).shape


# In[93]:


labels = ['voting','spam','volcanoes']
id3 = [0.985,0.648,0.743]
lr = [0.959,0.605,0.635]
nb = [0.94,0.635,0.61]
glad = [0.988,0.689,0.765]

x = np.arange(len(labels))  # the label locations
width = 0.35 

ig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, id3, width/3, label='ID3')
rects2 = ax.bar(x - width/6, lr, width/3, label='LogR')
rects3 = ax.bar(x + width/6, nb, width/3, label='NaiveB', color = 'purple')
rects4 = ax.bar(x + width/2, glad, width/3, label='GLAD', color = 'red')

ax.set_ylabel('Accuracy')
ax.set_title('Accurcy comparesion')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


# In[ ]:




