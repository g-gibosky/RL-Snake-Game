#!/usr/bin/env python
# coding: utf-8

# In[14]:


# A proxima célula le a função valor salva dos treinamentos anteriores e plota ela
# Assim, podemos avaliar como se comportam

import numpy as np
import pickle
import os
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import pandas as pd


def process_and_plot(data, file, current_path):
    if current_path == "reward_state_actions":
        title = "Reward"
    elif current_path == "reward_state_actions":
        title = "Steps"
    else:
        title = "loops"
    heatmap = np.zeros((19, 19))  # Assuming coordinates range from 0 to 9

    for (coords, _), value in data.items():
        x, y = coords
        heatmap[x+9][y+9] += value  # Summing up values; you can also use np.mean() for averaging

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(heatmap, cmap='crest', fmt=".2f")
    ax.invert_yaxis()
    # sns.heatmap(heatmap, cmap='viridis', fmt=".2f",  annot=True,)
    plt.title(f'{title}: {file}')
    ax.set_xticklabels([-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9])
    ax.set_yticklabels([-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9])
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(f"data\\Q_learning\\{current_path}\\graphs\\graph_{file}.jpeg")
    # plt.show()

path = 'data\\Q_learning\\'

# loop_files = os.listdir(f"{path}loop_steps")
# # print(loop_files)
# steps = []
# loop_ratios = []
# for file in loop_files:
#     with open(f"{path}loop_steps\\{file}", 'rb') as fp:
#         step, loop_ratio =  pickle.load(fp)
#         steps.append(step)
#         loop_ratios.append(loop_ratio)


# current_path = "reward_state_actions"
# reward_state_path = f"{path}{current_path}\\"
# print(f"reward_state_path: {reward_state_path}")
# reward_state_action = os.listdir(reward_state_path)
# print(f"Files: {reward_state_action}")
# for file in reward_state_action[1:]:
#     print(file)
#     with open(f"{reward_state_path}{file}", 'rb') as fp:
#         value_function = pickle.load(fp)
#         process_and_plot(value_function, file, current_path)

current_path = "steps_state_actions"
steps_state_path = f"{path}{current_path}\\"
steps_state_action = os.listdir(steps_state_path)
for file in steps_state_action[1:]:
    with open(f"{steps_state_path}{file}", 'rb') as fp:
        value_function = pickle.load(fp)
        process_and_plot(value_function, file, current_path)