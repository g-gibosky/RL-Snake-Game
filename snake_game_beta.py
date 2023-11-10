import pygame
import random
import time

# ESTADO É A DISTANCIA ENTRE A CABECA E A COMIDA
# DADA A POLITICA ALEATORIA QUAL A MEDIA DE REFORCO QUE ELE RECEBE


# Constants for the game grid
GRID_SIZE = 10
GRID_WIDTH = 10
GRID_HEIGHT = 15

# Define actions
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)  # Color for the snake's head

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((GRID_WIDTH * GRID_SIZE, GRID_HEIGHT * GRID_SIZE))
        pygame.display.set_caption("Snake Game - RL")
        self.delay_time = 0.0000001
        self.reset()

    def reset(self):
        self.snake = [(5, 5)]
        self.food = self.initialize_food()

    def initialize_food(self):
        empty_cells = [(x, y) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT) if (x, y) not in self.snake]
        return random.choice(empty_cells)

    def is_valid_move(self, action):
        new_head = (self.snake[0][0] + action[0], self.snake[0][1] + action[1])
        print("new_head[0]", new_head[0] < 0)
        print("new_head[0]", new_head[0] >= GRID_WIDTH)
        print("new_head[1]", new_head[1] < 0)
        print("new_head[1]", new_head[1] >= GRID_HEIGHT)
        print("new_head in ", new_head in self.snake) # erro esta aqui
        if (
            new_head[0] < 0
            or new_head[0] >= GRID_WIDTH
            or new_head[1] < 0
            or new_head[1] >= GRID_HEIGHT
            or new_head in self.snake
        ):
            return False

        return True

    def step(self, reward_shape):
        reward = 0
        # valid_actions = [action for action in ACTIONS if self.is_valid_move(action)]
        # action = random.choice(ACTIONS)
        action = self.get_desired_action()
        if not self.is_valid_move(action):
            print("Nao eh valido")
            return reward, False  # No reward and episode ends
        print("Action Selected", action)
        if self.snake[0] == self.food:
            self.food = self.initialize_food()
            self.snake.append((0, 0))
            reward = reward_shape[1]
        else:
            reward = -1
        #elif action != self.get_desired_action():
        #    reward = reward_shape[0]
        new_head = (self.snake[0][0] + action[0], self.snake[0][1] + action[1])
        self.snake.insert(0, new_head)
        self.snake.pop()

        return reward, True

    def render(self):
        self.screen.fill(WHITE)
        for i, segment in enumerate(self.snake):
            if i == 0:
                pygame.draw.rect(self.screen, BLUE, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            else:
                pygame.draw.rect(self.screen, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(self.screen, RED, (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.display.update()
        time.sleep(self.delay_time)
        
    # Optmization that make snake go only to the direction of the food, instead of choosing a random direction
    def get_desired_action(self):
        head = self.snake[0]
        food = self.food
        delta_x = food[0] - head[0]
        delta_y = food[1] - head[1]

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

class MonteCarloAgent:
    def __init__(self, game):
        self.game = game
        self.V = {}
        self.gamma = 0.9
        self.reward_shape = [-1, 99]

    def state_to_str(self, state):
        return str(state)

    def run_episode(self):
        
        episode_states = []
        episode_rewards = 0

        self.game.reset()

        while True:
            state = self.game.snake[0]
            state_str = self.state_to_str(state)
            reward, done = self.game.step(self.reward_shape)
            self.game.render()
            episode_rewards += reward
            episode_states.append(state)

            if not done:
                break

        for state in episode_states:
            state_str = self.state_to_str(state)
            if state_str not in self.V:
                self.V[state_str] = 0

            self.V[state_str] += episode_rewards

if __name__ == "__main__":
    game = SnakeGame()
    game.render()
    agent = MonteCarloAgent(game)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        agent.run_episode()
        print("Snake Size", game.snake)
        print("Finished Episode")
        game.render()

    pygame.quit()

    print("Learned State Values:")
    for state_str, value in agent.V.items():
        state = eval(state_str)
        print(f"State: {state}, Value: {value}")
