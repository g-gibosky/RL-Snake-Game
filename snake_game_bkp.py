import pygame
import random
import time

# Constants for the game grid
GRID_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 15

# Define actions
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)  # Color for the snake's head

# Initialize Pygame
pygame.init()

# Initialize the game window
screen = pygame.display.set_mode((GRID_WIDTH * GRID_SIZE, GRID_HEIGHT * GRID_SIZE))
pygame.display.set_caption("Snake Game")

# Snake initialization
snake = [(5, 5)]
snake_direction = (1, 0)  # Initial direction: right

# Function to initialize the food position
def initialize_food():
    return (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

# Initialize food
food = initialize_food()

# Initialize a dictionary for state values
V = {}

# Hyperparameters
gamma = 0.9  # Discount factor

# Helper function to convert a state to a string for dictionary indexing
def state_to_str(state):
    return str(state)

# Function to check if a move is valid
def is_valid_move(action, snake):
    new_head = (snake[0][0] + action[0], snake[0][1] + action[1])

    if (
        new_head[0] < 0
        or new_head[0] >= GRID_WIDTH
        or new_head[1] < 0
        or new_head[1] >= GRID_HEIGHT
        or new_head in snake
    ):
        return False

    return True

# Function to perform an episode
def run_episode():
    global food  # Declare food as a global variable to access it

    episode_states = []
    episode_rewards = 0
    add_segments = 0  # Number of segments to add when food is consumed

    # Perform an episode
    while True:
        valid_actions = [action for action in ACTIONS if is_valid_move(action, snake)]
        
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
                pygame.draw.rect(screen, BLUE, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            else:  # The snake's body
                pygame.draw.rect(screen, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(screen, RED, (food[0] * GRID_SIZE, food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
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
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Perform an episode
    run_episode()

# Quit the game
pygame.quit()

# Return the learned state values
print("Learned State Values:")
for state_str, value in V.items():
    state = eval(state_str)
    print(f"State: {state}, Value: {value}")
