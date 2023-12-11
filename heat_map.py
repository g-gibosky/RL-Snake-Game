# A proxima célula le a função valor salva dos treinamentos anteriores e plota ela
# Assim, podemos avaliar como se comportam

import numpy as np
import pickle
import os
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import pandas as pd


fig, axs = plt.subplots(1, 3, figsize=(20, 4))
path = 'data\\Q_learning\\Data2-20231210T212744Z-001\\Data2'
files = os.listdir('data\\Q_learning\\Data2-20231210T212744Z-001\\Data2')

for file in files:
  with open(path+'\\'+file, 'rb') as fp:
    value_function = pickle.load(fp)

  #value_function = agent.value_function
  matrix_values = np.zeros((19, 19))
  for k in list(value_function.keys()):
    x = int(k[0][1])
    y = int(k[1][0])

    matrix_values[x+9][y+9] = value_function[k]

  matrix_values[9][9] = 0

  sns.heatmap(matrix_values, ax = axs[i], cmap="crest")#, annot = True)
  axs[i].invert_yaxis()
  axs[i].set_xticks(range(19))
  axs[i].set_xticklabels([-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9])
  axs[i].set_yticks(range(19))
  axs[i].set_yticklabels([-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9])
  axs[i].set_title(f'Reward {n}')