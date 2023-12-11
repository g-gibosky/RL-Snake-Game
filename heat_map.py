# A proxima célula le a função valor salva dos treinamentos anteriores e plota ela
# Assim, podemos avaliar como se comportam

import numpy as np
import pickle
import os
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import pandas as pd


#essa função é para medir diferentes abordagens,
#com ela podemos ver quantos passos em média cada politca demora para chegar no objetivo
#além disso, podemos ver a % de vezes que o jogo entra em loop, ou seja, masi de 1000 passos


# def test_method(method, n_iter = 1000):
#   ep_len_total = 0
#   loop = 0
#   for i in range(n_iter):

#     ep_len = agent.run_episode(method = method, render = False, test_only = True)
#     ep_len_total +=  ep_len

#     if ep_len>=999:
#       loop+=1

#   ep_len_mean = ep_len_total/n_iter
#   prop_loop = loop/n_iter

#   return ep_len_mean, prop_loop

path = 'data\\Q_learning\\step_data\\'
loop_files = os.listdir(f"{path}loops")
print(loop_files)
for file in loop_files:
    with open(f"{path}\\loops\\{file}", 'rb') as fp:
        loop_values = pickle.load(fp)
        # loop_ratios, steps = 
    
# 'state_action_best_q_learning_1000_0_0_loop_interations


# fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# sns.lineplot(x = [100, 500, 1000, 5000], y = loop_ratios, ax = axs[0]).set_xscale("log")
# axs[0].set_title('Porcentagem de Loops Variando N0 (log scale)')
# axs[0].set_ylabel('% de Loops')

# sns.lineplot(x = [100, 500, 1000, 5000], y = steps, ax = axs[1]).set_xscale("log")
# axs[1].set_title('Número Médio de passos Variando N0 (log scale)')
# axs[1].set_ylabel('Tamanho médio do episódio')

# axs[0].set_xlabel('Parâmetro N0')
# axs[1].set_xlabel('Parâmetro N0')


# for file in files:
#   with open(path+'\\'+file, 'rb') as fp:
#     value_function = pickle.load(fp)

#   #value_function = agent.value_function
#   matrix_values = np.zeros((19, 19))
#   for k in list(value_function.keys()):
#     x = int(k[0][1])
#     y = int(k[1][0])

#     matrix_values[x+9][y+9] = value_function[k]

#   matrix_values[9][9] = 0

#   sns.heatmap(matrix_values, ax = axs[i], cmap="crest")#, annot = True)
#   axs[i].invert_yaxis()
#   axs[i].set_xticks(range(19))
#   axs[i].set_xticklabels([-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9])
#   axs[i].set_yticks(range(19))
#   axs[i].set_yticklabels([-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9])
#   axs[i].set_title(f'Reward {n}')