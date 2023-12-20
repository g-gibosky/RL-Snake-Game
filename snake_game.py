from collections import deque
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import pygame
import random
import numpy as np
import time
import pickle
import os
import sys
import seaborn as sns
sns.set()

# Constants for the game grid
GRID_SIZE = 40
GRID_WIDTH = 10
GRID_HEIGHT = 10

# Define actions
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)  # Color for the snake's head


class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.dense1 = layers.Dense(
            64, activation='relu', input_shape=(state_size,))
        self.dense2 = layers.Dense(64, activation='relu')
        self.prediction = layers.Dense(action_size, activation='linear')

    def call(self, inputs):

        # print(f"Call - Inputs: {inputs}")
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.prediction(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self):
      # Deixar aqui por que fiquei 10 min debugando por que nao achava o lerning rate
        self.state_size = 2
        self.action_size = 4
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
      # Instacinado tudo relacionado ao modelo
        self.memory = ReplayBuffer()
        self.model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)
        self.compile_models()
        self.update_target_model()
        self.loss = []

    def compile_models(self):
        loss_function = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=self.optimizer, loss=loss_function)
        self.target_model.compile(optimizer=self.optimizer, loss=loss_function)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.target_model.predict(
                        next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            self.loss.append(history.history['loss'][-1])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_state(self, state):
      normalized_state = np.array(state) / np.array([GRID_WIDTH, GRID_HEIGHT])
      return np.reshape(normalized_state, [1, 2])

    def act(self, state):
      # print(f"State in Act: {state}")
      if np.random.rand() <= self.epsilon:
          return random.randrange(self.action_size)
      state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
      q_values = self.model(state_tensor)
      return np.argmax(q_values[0])

    def train(self, raw_state, action, reward, raw_next_state, done):
      state = np.array([raw_state], dtype=np.float32)
      next_state = np.array([raw_next_state], dtype=np.float32)
      # print(f"State on train - raw_state: {raw_state} - raw_state shape: {state.shape}")
      # print(f"next_state on train - raw_next_state: {raw_next_state} - raw_state shape: {next_state.shape}")
      self.memory.add(state, action, reward, next_state, done)
      if self.memory.size() > batch_size:
          self.replay(batch_size)

    def test_act(self, state):
      state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
      # q_values = self.model(state_tensor)
      q_values = self.model.predict(state_tensor, verbose=0)
      return np.argmax(q_values[0])


class SnakeGame():
  def __init__(self, delay_time=0.00000001, reward_shape=(0, 99)):
      pygame.init()
      self.screen = pygame.display.set_mode(
          (GRID_WIDTH * GRID_SIZE, GRID_HEIGHT * GRID_SIZE))
      pygame.display.set_caption(
          "Snake Game - RL - Talasso, Ferreira, Gibosky")
      self.delay_time = delay_time
      self.reset()
      self.reward_shape = reward_shape
      pygame.font.init()  # you have to call this at the start,
      # if you want to use this module.
      self.game_font = pygame.font.SysFont("Arial", 20)
      self.step_num = 0

  def get_state(self):
    s = self.snake[0]
    f = self.food

    state_x = s[0] - f[0]
    state_y = s[1] - f[1]

    return (state_x, state_y)

  def reset(self):
      self.step_num = 0
      self.food = self.initialize_food()
      self.snake = self.initialize_snake(food=self.food)

  def reset_dqn(self):
      self.step_num = 0
      self.food = self.initialize_food()
      self.snake = self.initialize_snake(food=self.food)
      return self.snake

  def initialize_food(self):
      empty_cells = [(x, y) for x in range(GRID_WIDTH)
                     for y in range(GRID_HEIGHT)]
      return random.choice(empty_cells)

  def initialize_snake(self, food):
      empty_cells = [(x, y) for x in range(GRID_WIDTH)
                     for y in range(GRID_HEIGHT)]
      empty_cells.remove(food)
      return [random.choice(empty_cells)]

  def is_valid_move(self, action):

      new_head = (self.snake[0][0] + action[0], self.snake[0][1] + action[1])
      # print("new_head[0]", new_head[0] < 0)
      # print("new_head[0]", new_head[0] >= GRID_WIDTH)
      # print("new_head[1]", new_head[1] < 0)
      # print("new_head[1]", new_head[1] >= GRID_HEIGHT)
      # print("new_head in ", new_head in self.snake) # erro esta aqui
      if (
          new_head[0] < 0
          or new_head[0] >= GRID_WIDTH
          or new_head[1] < 0
          or new_head[1] >= GRID_HEIGHT
          or new_head in self.snake
      ):
          return False

      return True

  def step(self, reward_shape, method='random', value_function=None, N0=1, N=1,
           test_only=False, stochastic=False, stochastic_level=0.9, weights=None):

    reward = 0
    get = False
    # valid_actions = [action for action in ACTIONS if self.is_valid_move(action)]
    # action = random.choice(ACTIONS)
    action = self.get_desired_action(
        method=method, value_function=value_function, N0=N0, N=N, test_only=test_only, weights=weights)
    is_valid = self.is_valid_move(action)
    while not is_valid:
        # print("Nao e valido")
        action = self.get_desired_action(
            method='random', value_function=value_function, N0=N0, N=N, test_only=test_only, weights=weights)
        reward = reward_shape[0]
        is_valid = self.is_valid_move(action)
        # return reward, True  # No reward and episode ends
    # print("Action Selected", action)
    if self.snake[0] == self.food:
        self.food = self.initialize_food()
        self.snake.append((0, 0))
        reward = reward_shape[1]
        get = True
    else:
        reward = reward_shape[0]
    # elif action != self.get_desired_action():
    #    reward = reward_shape[0]

    if stochastic == 'all':  # se todo tabuleiro é estocastico
      # o com essa probabilidade, uma ação aleatoria é tomada
      if np.random.uniform(0, 1, 1)[0] < stochastic_level:

        action = self.get_desired_action(
            method='random', value_function=value_function, N0=N0, N=N, test_only=test_only, weights=weights)
        is_valid = self.is_valid_move(action)
        while not is_valid:
            # print("Nao e valido")
            action = self.get_desired_action(
                method='random', value_function=value_function, N0=N0, N=N, test_only=test_only, weights=weights)
            reward = reward_shape[0]
            is_valid = self.is_valid_move(action)

    if stochastic == 'up':  # se a parte de cima do tabuleiro é estocastico

      if self.snake[0][0] <= 4:
        # o com essa probabilidade, uma ação aleatoria é tomada
        if np.random.uniform(0, 1, 1)[0] < stochastic_level:

          action = self.get_desired_action(
              method='random', value_function=value_function, N0=N0, N=N, test_only=test_only, weights=weights)
          is_valid = self.is_valid_move(action)
          while not is_valid:
              # print("Nao e valido")
              action = self.get_desired_action(
                  method='random', value_function=value_function, N0=N0, N=N, test_only=test_only, weights=weights)
              reward = reward_shape[0]
              is_valid = self.is_valid_move(action)

    if stochastic == 'type2':  # se a parte de cima do tabuleiro é estocastico

      state = self.get_state()

      if state[0] == 0 or state[1] == 0:  # se estiver na mesma "reta" que a fruta
        # o com essa probabilidade, uma ação aleatoria é tomada
        if np.random.uniform(0, 1, 1)[0] < stochastic_level:

          action = self.get_desired_action(
              method='random', value_function=value_function, N0=N0, N=N, test_only=test_only, weights=weights)
          is_valid = self.is_valid_move(action)
          while not is_valid:
              # print("Nao e valido")
              action = self.get_desired_action(
                  method='random', value_function=value_function, N0=N0, N=N, test_only=test_only, weights=weights)
              reward = reward_shape[0]
              is_valid = self.is_valid_move(action)

    new_head = (self.snake[0][0] + action[0], self.snake[0][1] + action[1])

    self.snake[0] = new_head
    # self.snake.insert(0, new_head)
    # self.snake.pop()

    return reward, True, get, action

  def step_dqn(self, action):
    actual_action = ACTIONS[action]

    new_head = (self.snake[0][0] + actual_action[0],
                self.snake[0][1] + actual_action[1])

    if (new_head in self.snake or
        new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
        # Negative reward for losing, True for game over
        return self.reward_shape[0], self.get_state(), True

    # Check if the snake has eaten the food
    if new_head == self.food:
        self.snake.insert(0, new_head)  # Add new head to the snake
        self.food = self.initialize_food()  # Generate new food
        # Positive reward for eating food
        return self.reward_shape[1], self.get_state(), False

    # Normal movement (no food eaten)
    self.snake.insert(0, new_head)  # Add new head to the snake
    self.snake.pop()  # Remove the last segment of the snake

    # # Calculate a reward based on proximity to food (can be more complex)
    # distance_to_food = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
    reward = self.reward_shape[0]

    # Return reward, new state, and False for game not over
    return reward, self.get_state(), False

  def step_q(self, action):
      new_head = (self.snake[0][0] + action[0], self.snake[0][1] + action[1])

      if (new_head in self.snake) or (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
          return self.reward_shape[0], self.get_state(), False

      if new_head == self.food:
          self.food = self.initialize_food()
          return self.reward_shape[1], self.get_state(), True

      self.snake[0] = new_head

      distance_to_food = abs(
          new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
      # alternative reward value: reward = -distance_to_food
      reward = self.reward_shape[0]

      # Return reward, new state, and False for game not over
      return reward, self.get_state(), False

  def render(self, agent, render=True):

      if not render:
        return None

      self.screen.fill(WHITE)
      for i, segment in enumerate(self.snake):
          if i == 0:
              pygame.draw.rect(
                  self.screen, BLUE, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
          else:
              pygame.draw.rect(
                  self.screen, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
      pygame.draw.rect(
          self.screen, RED, (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
      txtsurf = self.game_font.render(
          f"Step: {self.step_num}", True, (0,0,0))
      self.screen.blit(txtsurf, (0, 0))
      txtsurf = self.game_font.render(
          f"Epi Size: {agent.epi_size} - Stocast: {agent.stochastic} - Stocast %: {agent.stochastic_percentage}", True, GREEN)
      self.screen.blit(txtsurf, (0, 25))
      txtsurf = self.game_font.render(
          f"Reward: {self.reward_shape[0]} - {self.reward_shape[1]}", True, BLUE)
      self.screen.blit(txtsurf, (0, 50))
      pygame.display.flip()
      pygame.display.update()
      time.sleep(self.delay_time)

      # #to run in Colab:
      # #convert image so it can be displayed in OpenCV
      # view = pygame.surfarray.array3d(self.screen)
      # img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
      # cv2_imshow(img_bgr)
      # time.sleep(self.delay_time)
      # output.clear()

  # Optmization that make snake go only to the direction of the food, instead of choosing a random direction
  def get_desired_action(self, method='random', value_function=None, N0=1, N=1, test_only=False, weights=None):
      head = self.snake[0]
      food = self.food
      delta_x = food[0] - head[0]
      delta_y = food[1] - head[1]

      if method == 'best':
        if delta_x > 0:
            return (1, 0)  # Right
        elif delta_x < 0:
            return (-1, 0)  # Left
        elif delta_y > 0:
            return (0, 1)  # Down
        elif delta_y < 0:
            return (0, -1)  # Up
        else:
            return (0, 0)  # No movement

      # choose a random action
      if method == 'random':
        possible_actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        idx = np.random.choice([0, 1, 2, 3])
        return possible_actions[idx]

      if method == 'value':
        # print(N0)
        epsilon = N0/(N0+N)

        if test_only:
          epsilon = 0.01

        if np.random.uniform(0, 1, 1)[0] < epsilon:
          possible_actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
          idx = np.random.choice([0, 1, 2, 3])
          return possible_actions[idx]

        state = self.get_state()
        possible_actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        max_value = -np.inf
        for a in possible_actions:

          future_state = (state[0] + a[0], state[1] + a[1])
          if self.is_valid_move(a):
            # print(future_state,state,  self.snake, self.food, value_function[str(future_state)])
            if value_function[str(future_state)] >= max_value:
              f = future_state
              v = value_function[str(future_state)]
              max_value = value_function[str(future_state)]
              action = a

      if method == 'linear_aproximator':
        # print(N0)
        epsilon = N0/(N0+N)

        if test_only:
          epsilon = 0.01

        if np.random.uniform(0, 1, 1)[0] < epsilon:
          possible_actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
          idx = np.random.choice([0, 1, 2, 3])
          return possible_actions[idx]

        state = self.get_state()
        possible_actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        max_value = -np.inf
        for a in possible_actions:

          future_state = (state[0] + a[0], state[1] + a[1])
          if self.is_valid_move(a):
            # print(future_state,state,  self.snake, self.food, value_function[str(future_state)])

            features = [0, 0]  # getting the features to make the decision
            features[0] = abs(future_state[0])
            features[1] = abs(future_state[1])

            predicted_future_value = weights[0]*features[0] + \
                weights[1]*features[1] + weights[2]  # linear function
            if predicted_future_value >= max_value:
              f = future_state
              v = predicted_future_value
              max_value = predicted_future_value
              action = a

        # print('--', v, f, action)
        return action


class RL_Agent(SnakeGame):
    def __init__(self, game, alpha=0.1, gamma=0.9, epsilon=1.0, decay=0.995, lambda_s=0.9, stochastic=0, stochastic_percentage=0.1):
        self.game = game
        self.q_table = {}  # Initialize Q-table, dict of tuples
        self.e_table = {}  # table for elegibility trace on sarsa(lambda)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.lambda_s = lambda_s
        self.decision_explore_exploit = []
        self.step_per_episode = []
        self.value_function = self.create_value_function()

        self.weights = np.zeros(3)  # two features one bias
        self.loss = []
        self.epi_size = 0
        self.stochastic = stochastic  # 0 not stochastic, 1 partialy stochastic
        self.stochastic_percentage = stochastic_percentage
        if stochastic == 1:
          self.stochastic_quadrant = self.define_stochastic_quadrant(
              stochastic_percentage)
          print("Quadrant", self.stochastic_quadrant)

    def define_stochastic_quadrant(self, percentage):
      # Up to percentage of the board is stochastic
      quadrant_width = int(GRID_WIDTH * percentage)
      quadrant_height = int(GRID_HEIGHT * percentage)
      return (0, 0, quadrant_width, quadrant_height)

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0)

    def get_e_value(self, state, action):
        return self.e_table.get((state, action), 0)

    def choose_action(self, state, choose_random=False, epsilon=None):

        if epsilon is None:
          epsilon = self.epsilon

        if choose_random:  # If agent is stuck on edge
          self.decision_explore_exploit.append(0)
          return random.choice(ACTIONS)

        # Explore or exploit
        if np.random.rand() < epsilon:
            self.decision_explore_exploit.append(0)  # Eploring
            return random.choice(ACTIONS)
        else:
            self.decision_explore_exploit.append(1)  # Exploiting
            q_values = [self.get_q_value(state, a) for a in ACTIONS]
            max_q = max(q_values)
            actions_with_max_q = [
                a for a in ACTIONS if self.get_q_value(state, a) == max_q]
            return random.choice(actions_with_max_q)

    def create_value_function(self):
      global GRID_HEIGHT
      global GRID_WIDTH
      value = {}

      for x in range(-GRID_WIDTH+1, GRID_WIDTH):
        for y in range(-GRID_HEIGHT+1, GRID_HEIGHT):

          value[f'({x}, {y})'] = 0

      value[str((0, 0))] = np.inf  # final state
      return value

    def is_in_stochastic_quadrant(self, x, y):
      x_start, y_start, qw, qh = self.stochastic_quadrant
      return x_start <= x < x_start + qw and y_start <= y < y_start + qh

    def train(self, episodes, sarsa):
      for episode in range(episodes):
        self.game.reset()
        state = self.game.get_state()
        done = False
        counter = 0
        while not done:
          counter += 1
          # Check if is a stochastic env or not and take action
          if self.stochastic == 1 and self.is_in_stochastic_quadrant(self.game.snake[0][0], self.game.snake[0][1]):
            action = actual_action = random.choice(ACTIONS)
          else:
            action = self.choose_action(state)

          # Check if action is valid
          is_valid = self.game.is_valid_move(action)
          while not is_valid:
            action = self.choose_action(state, choose_random=True)
            is_valid = self.game.is_valid_move(action)
          reward, next_state, done = self.game.step_q(action)
          if sarsa == True:
              # after getting the new state, I need to compute ne next action
              next_action = self.choose_action(next_state)
              self.update_q_table(state, action, reward,
                                  next_state, next_action)
          else:
            self.update_q_table(state, action, reward, next_state)
          state = next_state
        self.step_per_episode.append(counter)
        self.epsilon *= self.decay

    def train_sarsa_lambda(self, episodes):
        possible_actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for episode in range(episodes):
          self.game.reset()
          state = self.game.get_state()
          done = False
          counter = 0
          while not done:
              counter += 1
              action = self.choose_action(state)  # A
              is_valid = self.game.is_valid_move(action)
              while not is_valid:
                action = self.choose_action(state, choose_random=True)
                is_valid = self.game.is_valid_move(action)
              reward, next_state, done = self.game.step_q(action)  # R, S'
              next_action = self.choose_action(next_state)  # A'
              delta = self.get_delta(
                  state, action, reward, next_state, next_action)
              current_e = self.get_e_table(state, action)
              self.update_only_e_table(state, action, current_e)

             # updating qtable and etable for all states/actions
              for x in range(-GRID_WIDTH+1, GRID_WIDTH):
                for y in range(-GRID_HEIGHT+1, GRID_HEIGHT):
                    state_ = [(x, y)]
                    for actions in possible_actions:
                        reward_, next_state_, done_ = self.game.step_q(actions)
                        self.update_q_table(state_[0], actions, reward_, delta)

              state = next_state
              action = next_action

          self.step_per_episode.append(counter)
          self.epsilon *= self.decay

    def run_episode(self, sarsa, sarsa_lambda=False):  # to test a trained agent

      epsilon = 0  # to test the policy, without random effects

      self.game.reset()
      state = self.game.get_state()
      done = False
      ep_len = 0

      while not done:
          ep_len += 1

          if sarsa == True:
            action = self.choose_action(state, epsilon=0)
          else:
            action = self.choose_action(state)

          reward, next_state, done = self.game.step_q(action)
          self.game.step_num = ep_len
          self.game.render(self)

          if sarsa == True:
              # after getting the new state, I need to compute ne next action
              next_action = self.choose_action(next_state, epsilon=0)
              self.update_q_table(state, action, reward,
                                  next_state, next_action)
              # print('entrei no sarsa')
          else:
            if sarsa_lambda:
              next_action = self.choose_action(next_state, epsilon=0)
              self.update_q_table(state, action, reward, self.get_delta(
                  state, action, reward, next_state, next_action))
            else:
              self.update_q_table(state, action, reward, next_state)
              # print('entrei no qlearning')
          state = next_state

      return ep_len

    def linear_aproximator(self, state, V_sampled, epochs=1, lr=0.00001, n=1):

      # V_sampled = 2*state[0] + 5*state[1] + 10

      features = [0, 0]
      features[0] = abs(state[0])
      features[1] = abs(state[1])

      for i in range(epochs):

        # Linear equation: Y = a_1*x_1 + a_2*x_2 + b
        V_pred = self.weights[0]*features[0] + \
            self.weights[1]*features[1] + self.weights[2]
        # print(state, V_sampled, V_pred)

        D0 = (features[0] * (V_pred - V_sampled))  # gradient
        self.weights[0] = self.weights[0] - (lr * D0)  # update

        D1 = (features[1] * (V_pred - V_sampled))  # gradient
        self.weights[1] = self.weights[1] - (lr * D1)  # update

        D2 = (V_pred - V_sampled)  # gradient
        self.weights[2] = self.weights[2] - (lr * D2)  # update

      loss = 0.5 * (V_pred - V_sampled)**2

      # print(self.weights)
      self.loss.append(loss)


def test_agent(agent, game, episodes):
  ep_lens = []
  max_steps = 10
  for episode in range(episodes):
      state = game.reset_dqn()
      print(f"Current state: {state}")
      total_reward = 0
      done = False
      ep_len = 0
      while not done and ep_len < max_steps:
          action_index = agent.test_act(state)
          reward, next_state, done = game.step_dqn(action_index)
          # game.render(episode)
          print(f"State: {state} - Reward: {reward}, next_state: {next_state}")
          state = next_state
          # print(f"Reward: {reward}")
          total_reward += reward
          ep_len += 1

          if done:
              print(f"Episode: {episode+1}, Total Reward: {total_reward}")
              ep_lens.append(ep_len)
              break
  return ep_lens


class QLearning(RL_Agent):
  def update_q_table(self, state, action, reward, next_state):
      max_q_next = max([self.get_q_value(next_state, a) for a in ACTIONS])
      current_q = self.get_q_value(state, action)
      new_q = current_q + self.alpha * \
          (reward + self.gamma * max_q_next - current_q)
      self.q_table[(state, action)] = new_q
      self.value_function[str(state)] = new_q

      self.linear_aproximator(state, new_q)


def run_q_tests(epis_size, stochastic, test_game, percentage=0.0):
  test_game.current_episode_max = epis_size
  q_learning = QLearning(test_game, stochastic=stochastic,
                         stochastic_percentage=percentage)
  print(
      f"Treinamento em ambiente: { 'Partialy' if q_learning.stochastic == 1 else 'non-stochastic'}")
  q_learning.epi_size = epis_size
  q_learning.train(epis_size, sarsa=False)
  ep_len_mean, prop_loop = test_q_policy(q_learning)
  plot_steps(q_learning.step_per_episode, stochastic,epis_size,ep_len_mean, percentage)
  print(f"ep_len_mean: {ep_len_mean}")


def test_q_policy(agent, n_iter=1000, sarsa=False, sarsa_lambda=False):
    ep_len_total = 0
    loop = 0
    for _ in range(n_iter):
        ep_len = agent.run_episode(sarsa=sarsa, sarsa_lambda=sarsa_lambda)
        ep_len_total += ep_len

        if ep_len >= 999:
          loop += 1

    ep_len_mean = ep_len_total/n_iter
    prop_loop = loop/n_iter

    return ep_len_mean, prop_loop

def plot_steps(step_per_episodes, stochastic,epis_size,ep_len_mean, stochastic_percentage=0):
#    Calculate the mean of the stepses
   mean_step = np.mean(step_per_episodes)

    # Plot the mean steps as a horizontal line

   plt.figure(figsize=(10, 6))
   plt.plot(step_per_episodes, label='Steps per Episode')
   plt.title(f'Steps per Episode: { f"Partialy Stochastic {stochastic_percentage}" if stochastic == 1 else "Non-Stochastic"}')
   plt.xlabel('Episode Number')
   plt.ylabel('Steps')
   plt.axhline(y=ep_len_mean, color='r', linestyle='--', label=f'Mean Steps Best: {ep_len_mean:.2f}')
   plt.legend()
   plt.grid(True)
  #  plt.show()
   plt.savefig(f'./data/state_action_best_q_learning_{epis_size}_{stochastic}_{stochastic_percentage}_runs.png')

def run_q_tests_reward(epis_size, test_game, reward):
 q_learning = QLearning(test_game)
 q_learning.train(epis_size, sarsa = False)
 ep_len_mean, prop_loop = test_q_policy(q_learning)
 plot_steps_reward(q_learning.step_per_episode, ep_len_mean, reward)

def plot_steps_reward(step_per_episodes, ep_len_mean, reward_shape):

   plt.figure(figsize=(10, 6))
   plt.plot(step_per_episodes, label='Steps per Episode')
   plt.title(f'Steps per Episode - Incentivo: {reward_shape[1]} - Punição: {reward_shape[0]}')
   plt.xlabel('Episode Number')
   plt.ylabel('Steps')
   plt.axhline(y=ep_len_mean, color='r', linestyle='--', label=f'Mean Steps Best: {ep_len_mean:.2f}')
   plt.legend()
   plt.grid(True)
  #  plt.show()
   plt.savefig(f'./data/state_action_best_q_learning_{reward_shape[1]}_{reward_shape[0]}_runs.png')
   
test_game = SnakeGame()
episode_size = [100, 500, 1000, 5000]
percentages = [0.2, 0.5, 0.9]
percentages = [0.2, 0.5, 0.9]
# run_q_tests(5000, 0, test_game)
for stochastic in range(0, 2):
 for episodes in episode_size:
   print(f"stochastic: {stochastic} - Episodes: {episodes}")
   if stochastic == 1:
     for percentage in [0.2, 0.5, 0.9]:
       run_q_tests(episodes, stochastic, test_game, percentage)
   else:
     run_q_tests(episodes, stochastic, test_game)
     
     
episode_size = [100, 500, 1000, 5000]
reward_shape = [(-1, 99), (0, 99), (-10,99)]
for reward in reward_shape:
 test_game = SnakeGame(reward_shape=reward)
 for episodes in episode_size:
     run_q_tests_reward(episodes, test_game, reward)


# agent = DQNAgent()
# batch_size = 32  # Define an appropriate batch size
# game = SnakeGame()
# total_episodes = 10
# max_steps_per_episode = 10

# for episode in range(total_episodes):
#     print(f"Current Episode: {episode}")
#     state = game.get_state()  # Get initial state
#     total_reward = 0

#     for _ in range(max_steps_per_episode):
#         action_index = agent.act(state)
#         # action = ACTIONS[action_index]
#         # print(action)
#         # print(f"Current Step: {t}")
#         reward, next_state, done = game.step_dqn(action_index)
#         agent.train(state, action_index, reward, next_state, done)

#         state = next_state
#         total_reward += reward

#         if done:
#           print(done)
#           break

# test_episodes = 20
# game_test = SnakeGame()
# game_test.render()
# ep_lens = test_agent(agent, game_test, test_episodes)
# test_agent(agent, game, test_episodes)
# print(np.mean(ep_lens))
