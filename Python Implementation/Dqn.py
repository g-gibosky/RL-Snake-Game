# AI Dependencies
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
import numpy as np
from .snake_game_base import SnakeGame

# Trocar para .env

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
        self.state_size = 2
        self.action_size = 4
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
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
      self.memory.add(state, action, reward, next_state, done)
      if self.memory.size() > batch_size:
          self.replay(batch_size)

    def test_act(self, state):
      state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
      # q_values = self.model(state_tensor)
      q_values = self.model.predict(state_tensor, verbose=0)
      return np.argmax(q_values[0])
  
    def run(self):
        game = SnakeGame()
        total_episodes = 10
        max_steps_per_episode = 10

        for episode in range(total_episodes):
            print(f"Current Episode: {episode}")
            state = game.get_state()  # Get initial state
            total_reward = 0

            for t in range(max_steps_per_episode):
                action_index = self.act(state)
                # action = ACTIONS[action_index]
                # print(action)
                print(f"Current Step: {t}")
                reward, next_state, done = game.step_dqn(action_index)
                self.train(state, action_index, reward, next_state, done)

                state = next_state
                total_reward += reward

                if done:
                    print(done)
                    break

agent = DQNAgent()
agent.run()