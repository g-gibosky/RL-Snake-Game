import pygame
import random

# Initialize Pygame
pygame.init()

# Constants for the game
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
SNAKE_SIZE = 20
FOOD_SIZE = 20
SNAKE_SPEED = 20

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [((SCREEN_WIDTH // 2), (SCREEN_HEIGHT // 2))]
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])  # Changed to tuples
        self.color = GREEN

    def turn(self, point):
        if self.length > 1 and (point[0] == -self.direction[0] and point[1] == -self.direction[1]):
            return
        else:
            self.direction = point

    def get_head_position(self):
        return self.positions[0]

    def move(self):
        cur = self.get_head_position()
        x, y = self.direction
        new = (((cur[0] + (x * SNAKE_SIZE)) % SCREEN_WIDTH), (cur[1] + (y * SNAKE_SIZE)) % SCREEN_HEIGHT)
        if len(self.positions) > 2 and new in self.positions[2:]:
            self.reset()
        else:
            self.positions.insert(0, new)
            if len(self.positions) > self.length:
                self.positions.pop()

    def reset(self):
        self.length = 1
        self.positions = [((SCREEN_WIDTH // 2), (SCREEN_HEIGHT // 2))]
        self.direction = random.choice([pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT])

    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0], p[1]), (SNAKE_SIZE, SNAKE_SIZE))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, WHITE, r, 1)

    def handle_keys(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.turn((0, -1))
                elif event.key == pygame.K_DOWN:
                    self.turn((0, 1))
                elif event.key == pygame.K_LEFT:
                    self.turn((-1, 0))
                elif event.key == pygame.K_RIGHT:
                    self.turn((1, 0))

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = RED
        self.randomize_position()

    def randomize_position(self):
        self.position = (random.randint(0, SCREEN_WIDTH//SNAKE_SIZE - 1) * SNAKE_SIZE,
                         random.randint(0, SCREEN_HEIGHT//SNAKE_SIZE - 1) * SNAKE_SIZE)

    def draw(self, surface):
        r = pygame.Rect((self.position[0], self.position[1]), (FOOD_SIZE, FOOD_SIZE))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, WHITE, r, 1)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.snake = Snake()
        self.food = Food()

    def run(self):
        while True:
            self.screen.fill((0, 0, 0))
            self.snake.handle_keys()
            self.snake.move()
            if self.snake.get_head_position() == self.food.position:
                self.snake.length += 1
                self.food.randomize_position()
            self.snake.draw(self.screen)
            self.food.draw(self.screen)
            pygame.display.update()
            self.clock.tick(SNAKE_SPEED)

# Main execution
if __name__ == "__main__":
    game = Game()
    game.run()
