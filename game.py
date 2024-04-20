import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the screen
GRID_WIDTH = 13
GRID_HEIGHT = 13
BLOCK_SIZE = 30
SCREEN_WIDTH = GRID_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * BLOCK_SIZE
pygame.display.set_caption("Snake Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
UP_LEFT = (-1, -1)
UP_RIGHT = (1, -1)
DOWN_LEFT = (-1, 1)
DOWN_RIGHT = (1, 1)

def distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p2[1] - p1[1])

def vector_sum(p1, p2):
    return p1[0] + p2[0], p1[1] + p2[1]

class SnakeGame:
    def __init__(self, model, food_pos):
        self.end = False
        self.snake = [(random.randint(0, GRID_WIDTH) // 2 * BLOCK_SIZE, random.randint(0, GRID_HEIGHT) // 2 * BLOCK_SIZE)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.clock = pygame.time.Clock()
        self.model = model
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.food_positions = food_pos
        self.food_position_iter = iter(self.food_positions) if self.food_positions is not None else None
        self.food = self.get_food_pos()

    def __convert_to_abs_direction(self, rel_direction):
        current_dir = self.direction
        if current_dir is None or current_dir == UP:
            return rel_direction
        if current_dir == DOWN:
            abs_direction = {
                UP: DOWN,
                DOWN: UP,
                LEFT: RIGHT,
                RIGHT: LEFT
            }
        elif current_dir == LEFT:
            abs_direction = { UP: LEFT, DOWN: RIGHT, LEFT: DOWN, RIGHT: UP }
        else:
            abs_direction = { UP: RIGHT, DOWN: LEFT, LEFT: UP, RIGHT: DOWN }

        abs_direction[UP_LEFT] = vector_sum(abs_direction[UP], abs_direction[LEFT])
        abs_direction[UP_RIGHT] = vector_sum(abs_direction[UP], abs_direction[RIGHT])
        abs_direction[DOWN_LEFT] = vector_sum(abs_direction[DOWN], abs_direction[LEFT])
        abs_direction[DOWN_RIGHT] = vector_sum(abs_direction[DOWN], abs_direction[RIGHT])

        return abs_direction[rel_direction]

    def get_food_pos(self):
        if self.food_position_iter is None:
            return random.randint(0, GRID_WIDTH - 1) * BLOCK_SIZE, random.randint(0, GRID_HEIGHT - 1) * BLOCK_SIZE
        else:
            next_food = next(self.food_position_iter, None)
            if next_food is not None:
                return next_food[0] * BLOCK_SIZE, next_food[1] * BLOCK_SIZE
            else:
                return (random.randint(0, GRID_WIDTH - 1) * BLOCK_SIZE,
                        random.randint(0, GRID_HEIGHT - 1) * BLOCK_SIZE) 

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def is_obstacle(self, point):
        return (point[0] not in range(0, GRID_WIDTH * BLOCK_SIZE) 
                or point[1] not in range(0, GRID_HEIGHT * BLOCK_SIZE)
                or (point[0], point[1]) in self.snake)

    def is_wall(self, point):
        return (point[0] not in range(0, GRID_WIDTH * BLOCK_SIZE) 
                or point[1] not in range(0, GRID_HEIGHT * BLOCK_SIZE))

    def is_body(self, point):
        return (point[0], point[1]) in self.snake

    def move_snake(self):
        cur = self.snake[0]
        x, y = self.direction
        new = cur[0] + (x * BLOCK_SIZE), cur[1] + (y * BLOCK_SIZE)

        if self.is_obstacle(new):
            return False
        else:
            self.snake.insert(0, new)
            if self.snake[0] == self.food:
                self.food = self.get_food_pos()
            else:
                self.snake.pop()
        return True

    def get_input(self):
        head = self.snake[0]

        grid_head = (head[0] // BLOCK_SIZE, head[1] // BLOCK_SIZE)
        grid_food = self.food[0] // BLOCK_SIZE, self.food[1] // BLOCK_SIZE
        
        directions = [
                self.__convert_to_abs_direction(UP),
                self.__convert_to_abs_direction(LEFT),
                self.__convert_to_abs_direction(RIGHT),
            ]

        obstacles = []

        head_to_food = [grid_food[0] - grid_head[0], grid_food[1] - grid_head[1]]

        if head_to_food[0] != 0:
            head_to_food[0] //= abs(head_to_food[0])

        if head_to_food[1] != 0:
            head_to_food[1] //= abs(head_to_food[1])
        
        for direction in directions:
            tile = (grid_head[0] + direction[0], grid_head[1] + direction[1])
            if self.is_obstacle((tile[0] * BLOCK_SIZE, tile[1] * BLOCK_SIZE)):
                obstacles.append(1)
            else:
                obstacles.append(0)
        dir_to_food = None
        for direction in directions:
            if self.__convert_to_abs_direction(direction) == (head_to_food[0], head_to_food[1]):
                dir_to_food = direction
                break

        if dir_to_food is None:
            dir_to_food = (0, 0)

        single_dir_to_food = []
        if dir_to_food in [UP, UP_LEFT, UP_RIGHT]:
            single_dir_to_food.append(1)
        else:
            single_dir_to_food.append(0)

        if dir_to_food in [UP_LEFT, LEFT, DOWN_LEFT]:
            single_dir_to_food.append(1)
        else:
            single_dir_to_food.append(0)

        if dir_to_food in [UP_RIGHT, RIGHT, DOWN_RIGHT]:
            single_dir_to_food.append(1)
        else:
            single_dir_to_food.append(0)

        return obstacles + [*single_dir_to_food]

    def reset(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.food = self.get_food_pos()

    def draw(self):
        self.screen.fill(BLACK)
        for pos in self.snake:
            pygame.draw.rect(self.screen, WHITE, (pos[0], pos[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.screen, RED, (self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.update()

    def make_decision(self):
        inp = self.get_input()

        [up, left, right] = self.model.activate(inp)
        max_out = max(up, left, right)
        if max_out == up:
            self.direction = self.__convert_to_abs_direction(UP)
        elif max_out == left:
            self.direction = self.__convert_to_abs_direction(LEFT)
        else:
            self.direction = self.__convert_to_abs_direction(RIGHT)


    def run(self):
        while True:
            self.handle_events()
            if not self.end:
                if not self.move_snake():
                    self.end = True
                self.make_decision()
            self.draw()
            self.clock.tick(10)
