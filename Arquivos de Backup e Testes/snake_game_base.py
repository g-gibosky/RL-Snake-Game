import pandas as pd
import matplotlib.pyplot as plt
import pygame
import random
import cv2
import numpy as np
import time
import pickle
import os
import seaborn as sns
sns.set()

os.environ["SDL_VIDEODRIVER"] = "dummy"

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

"""#Ambiente e Agente"""


class SnakeGame():
  def __init__(self, delay_time=0.000005, reward_shape=(-1, 99)):
      pygame.init()
      self.screen = pygame.display.set_mode(
          (GRID_WIDTH * GRID_SIZE, GRID_HEIGHT * GRID_SIZE))
      pygame.display.set_caption(
          "Snake Game - RL - Talasso, Ferreira, Gibosky")
      self.delay_time = delay_time
      self.reset()
      self.reward_shape = reward_shape

  def get_state(self):
    s = self.snake[0]
    f = self.food

    state_x = s[0] - f[0]
    state_y = s[1] - f[1]

    return (state_x, state_y)

  def reset(self):
      self.food = self.initialize_food()
      self.snake = self.initialize_snake(food=self.food)

  def reset_dqn(self):
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
        return -10, self.get_state(), True

    # Check if the snake has eaten the food
    if new_head == self.food:
        self.snake.insert(0, new_head)  # Add new head to the snake
        self.food = self.initialize_food()  # Generate new food
        return 10, self.get_state(), False  # Positive reward for eating food

    # Normal movement (no food eaten)
    self.snake.insert(0, new_head)  # Add new head to the snake
    self.snake.pop()  # Remove the last segment of the snake

    # Calculate a reward based on proximity to food (can be more complex)
    distance_to_food = abs(
        new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
    reward = -distance_to_food

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

  def render(self, render=True):

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
      pygame.display.update()

      # to run in Colab:
      # convert image so it can be displayed in OpenCV
      view = pygame.surfarray.array3d(self.screen)
      img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
      cv2_imshow(img_bgr)
      time.sleep(self.delay_time)
      output.clear()

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
    
  def run_episode(self):
        global food  # Declare food as a global variable to access it

        episode_states = []
        episode_rewards = 0
        add_segments = 0  # Number of segments to add when food is consumed

        # Perform an episode
        while True:
            valid_actions = [
                action for action in ACTIONS if is_valid_move(action, snake)]

            if not valid_actions:
                break

            action = random.choice(valid_actions)

            # Move the snake
            new_head = (snake[0][0] + action[0], snake[0][1] + action[1])
            snake.insert(0, new_head)

            # Check for collision with food
            if snake[0] == food:
                reward = 1
                food = initialize_food()  # Reinitialize food position
                add_segments += 1  # Increase the number of segments to add

            # Check for collision with walls or itself
            if (
                snake[0][0] < 0
                or snake[0][0] >= GRID_WIDTH
                or snake[0][1] < 0
                or snake[0][1] >= GRID_HEIGHT
                or snake[0] in snake[1:]
            ):
                reward = -10
                break

                episode_rewards += reward
                episode_states.append(snake[0])

            # Remove tail segments if needed
            if add_segments > 0:
                add_segments -= 1
            else:
                snake.pop()

            # Draw the game grid
            screen.fill(WHITE)
            for i, segment in enumerate(snake):
                if i == 0:  # The snake's head
                    pygame.draw.rect(
                        screen, BLUE, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
                else:  # The snake's body
                    pygame.draw.rect(
                        screen, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(
                screen, RED, (food[0] * GRID_SIZE, food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            pygame.display.update()

            # Add a short delay to control the game speed (adjust as needed)
            time.sleep(0.1)

        # Calculate state values using Monte Carlo return
        for state in episode_states:
            state_str = state_to_str(state)
            if state_str not in V:
                V[state_str] = 0

            V[state_str] += episode_rewards

# Game loop
running = True
game = SnakeGame()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Perform an episode
    game.run_episode()

# Quit the game
pygame.quit()

# Return the learned state values
print("Learned State Values:")
for state_str, value in V.items():
    state = eval(state_str)
    print(f"State: {state}, Value: {value}")
