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

path = 'data\\Q_learning\\'

# loop_files = os.listdir(f"{path}loop_steps")
# print(loop_files)
# steps = []
# loop_ratios = []
# for file in loop_files:
#     with open(f"{path}loop_steps\\{file}", 'rb') as fp:
#         step, loop_ratio =  pickle.load(fp)
#         steps.append(step)
#         loop_ratios.append(loop_ratio)

# group_size = 4
# avg_steps = [sum(steps[i:i+group_size])/group_size for i in range(0, len(steps), group_size)]
# avg_loops = [sum(loop_ratios[i:i+group_size])/group_size for i in range(0, len(loop_ratios), group_size)]

# x_values = [100, 500, 1000, 5000]

# print(steps, loop_ratios)
# fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# sns.lineplot(x=x_values, y=avg_loops, ax=axs[0]).set_xscale("log")
# axs[0].set_title('Porcentagem de Loops Variando N0 (log scale)')
# axs[0].set_ylabel('% de Loops')
# axs[0].set_xlabel('Parâmetro N0')

# sns.lineplot(x=x_values, y=avg_steps, ax=axs[1]).set_xscale("log")
# axs[1].set_title('Número Médio de passos Variando N0 (log scale)')
# axs[1].set_ylabel('Tamanho médio do episódio')
# axs[1].set_xlabel('Parâmetro N0')

# plt.show(block=True)


def process_and_plot(data, file):
    heatmap = np.zeros((10, 10))  # Assuming coordinates range from 0 to 9

    for (coords, _), value in data.items():
        x, y = coords
        heatmap[x, y] += value  # Summing up values; you can also use np.mean() for averaging

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap, cmap='viridis', fmt=".2f")
    # sns.heatmap(heatmap, cmap='viridis', fmt=".2f",  annot=True,)
    plt.title(f'Reward: {file}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(f"graph_{file}.jpeg")
    plt.show()


# reward_state_path = f"{path}reward_state_actions\\"
# reward_state_action = os.listdir(reward_state_path)
# for file in reward_state_action:
#     print(file)
#     with open(f"{reward_state_path}{file}", 'rb') as fp:
#         value_function = pickle.load(fp)
#         process_and_plot(value_function, file)

steps_state_path = f"{path}steps_state_actions\\"
steps_state_action = os.listdir(steps_state_path)
for file in steps_state_action:
    print(file)
    with open(f"{steps_state_path}{file}", 'rb') as fp:
        value_function = pickle.load(fp)
        print(value_function)
        process_and_plot(value_function, file)