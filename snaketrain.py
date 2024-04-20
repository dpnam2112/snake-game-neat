from random import choice, randint, random, shuffle
import statistics

import neat

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
UP_LEFT = (-1, -1)
UP_RIGHT = (1, -1)
DOWN_LEFT = (-1, 1)
DOWN_RIGHT = (1, 1)

MAX_MOVES = 500

def distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p2[1] - p1[1])

def vector_sum(p1, p2):
    return p1[0] + p2[0], p1[1] + p2[1]

def random_food_positions(food_count, grid_width, grid_height):
    i = 0
    x_interval = int(grid_width / 3)
    y_interval = int(grid_height / 3)
    food_pos = []
    coefficients = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    count = 0

    while count < food_count:
        for cx, cy in coefficients:
            for _ in range(3):
                randx, randy = randint(cx * x_interval, (cx + 1) * x_interval), randint(cy * y_interval, (cy + 1) * y_interval)
                food_pos.append((randx, randy))
        count += 3
    
    shuffle(food_pos)
    return food_pos

class TrainingInstance:
    """Used to represent the game state. """
    def __init__(self, grid_width, grid_height, model, default_food_pos):
        """TrainingInstance's constructor. 

        Args:
            grid_width (int): number of columns of the board.
            grid_height (int): number of rows of the board.
            model: The function used by the training instance to make decision. 'model' must
            have 'activate' function that takes a list as its input, and return a vector as its 
            output.
        """

        self.fitness = 0
        self.direction = choice([UP, DOWN, LEFT, RIGHT])
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.model = model
        self.moves = 0
        init_x, init_y = [self.grid_width // 2, self.grid_height // 2]
        self.init_positions = [(init_x, init_y)]
        self.positions = self.init_positions[:]
        self.food_positions = default_food_pos
        self.food_position_iter = iter(self.food_positions) if self.food_positions else None
        self.food_pos = self.get_food_pos()
        self.total_moves = 0
        self.food_seq = []


    def reset(self):
        self.length = 1
        self.fitness = 0
        self.positions = self.init_positions[:]
        self._current_food = -1
        self.total_moves = 0
        self.food_pos = (randint(0, self.grid_width - 1), randint(0, self.grid_height - 1))
        self.positions = self.init_positions
        self.food_position_iter = iter(self.food_positions) if self.food_positions is not None else None
        self.direction = choice([UP, DOWN, LEFT, RIGHT])

    def get_fitness(self):
        return self.length

    def __get_head(self):
        return self.positions[-1]

    def __get_next_head(self):
        cur_head_x, cur_head_y = self.__get_head()
        return cur_head_x + self.direction[0], cur_head_y + self.direction[1]

    def __is_obstacle(self, pos):
        return (pos in self.positions 
                or pos[0] not in range(0, self.grid_width)
                or pos[1] not in range(0, self.grid_height))

    def __is_wall(self, pos):
        return (pos[0] not in range(0, self.grid_width)
                or pos[1] not in range(0, self.grid_height))

    def __is_body(self, pos):
        return pos in self.positions

    def __get_input(self):
        head = self.__get_head()

        walls = []
        to_body = []
        obstacles = []
        to_food = []

        directions = [
                self.__convert_to_abs_direction(UP),
                self.__convert_to_abs_direction(LEFT),
                self.__convert_to_abs_direction(RIGHT),
#                self.__convert_to_abs_direction(UP_LEFT),
#                self.__convert_to_abs_direction(UP_RIGHT),
#                self.__convert_to_abs_direction(DOWN_LEFT),
#                self.__convert_to_abs_direction(DOWN_RIGHT)
                ]

        head_to_food = [self.food_pos[0] - head[0], self.food_pos[1] - head[1]]

        if head_to_food[0] != 0:
            head_to_food[0] //= abs(head_to_food[0])

        if head_to_food[1] != 0:
            head_to_food[1] //= abs(head_to_food[1])

        dir_to_food = [0, 0]

        for direction in directions:
            tile = [head[0] + direction[0], head[1] + direction[1]]
            body = None

            obstacles.append(1 if self.__is_obstacle(tile) else 0)

        for direction in directions:
            abs_direction = self.__convert_to_abs_direction(direction)
            if abs_direction[0] == head_to_food[0] and abs_direction[1] == head_to_food[1]:
                dir_to_food = direction
                break

        single_dir_to_food = []
        if dir_to_food in [UP, UP_LEFT, UP_RIGHT]:
            single_dir_to_food.append(1)
        else:
            single_dir_to_food.append(0)

        if dir_to_food in [UP_LEFT, LEFT, DOWN_LEFT]:
            single_dir_to_food.append(1)
        else:
            single_dir_to_food.append(0)

        if single_dir_to_food in [UP_RIGHT, RIGHT, DOWN_RIGHT]:
            single_dir_to_food.append(1)
        else:
            single_dir_to_food.append(0)

        return obstacles + [*single_dir_to_food]

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
        if self.food_positions is None:
            return (randint(0, self.grid_width - 1), randint(0, self.grid_height - 1))
        else:
            next_food = next(self.food_position_iter, None)
            return next_food if next_food is not None else (randint(0, self.grid_width - 1), randint(0, self.grid_height - 1)) 

    def update(self):
        self.moves += 1

        if self.moves > 50:
            return False

        if self.__get_next_head() == self.food_pos:
            self.food_seq.append(self.food_pos)
            self.food_pos = self.get_food_pos()
            #self.fitness += 1
            self.length += 1
            self.total_moves += self.moves
            self.moves = 0

        prev_head = self.__get_head()
        cur_head = self.__get_next_head()

        self.positions.append(cur_head)
        if len(self.positions) > self.length:
            self.positions.pop(0)

        if (cur_head[0] not in range(0, self.grid_width) 
            or cur_head[1] not in range(0, self.grid_height)
            or cur_head in self.positions[:-1]):
            # the snake collides with walls or its body
            return False

        inp = self.__get_input()


        [up, left, right] = self.model.activate(inp)

        max_out = max(up, left, right)
        if max_out == up:
            rel_dir = UP
        elif max_out == left:
            rel_dir = LEFT
        else:
            rel_dir = RIGHT

        dir_to_food = [self.food_pos[0] - prev_head[0], self.food_pos[1] - prev_head[1]]

        if dir_to_food[0] != 0:
            dir_to_food[0] = dir_to_food[0] / abs(dir_to_food[0])
        if dir_to_food[1] != 0:
            dir_to_food[1] = dir_to_food[1] / abs(dir_to_food[1])

        self.direction = self.__convert_to_abs_direction(rel_dir)
        next_head = self.__get_next_head()
#        if distance(next_head, self.food_pos) < distance(self.__get_head(), self.food_pos):
#            self.fitness += 0.01 * self.length
        
        return True

class TrainingPlayground:
    """This class is used to build a decision-making model for the snake in the snake game."""
    def __init__(self, grid_width, grid_height, sim_times, default_food_positions):
        """Trainer's constructor.

        Args:
            sim_times (int): Number of times that each training instances are simulated.
            population (int): Number of population of each generation.
            grid_width (int), grid_height (int): The board's grid size.
            config (str): directory to the NEAT configuration file.
        """
        self.sim_times = sim_times
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_score = 0
        self.default_food_positions = default_food_positions
        self.best_genome = None

    def train(self, genomes, models):
        for genome, model in zip(genomes, models):
            training_instance = TrainingInstance(self.grid_width, self.grid_height, model, self.default_food_positions)
            self.train_instance(training_instance)
            genome.fitness = self.train_instance(training_instance)

    def train_instance(self, training_instance: TrainingInstance):
        scores = []
        for _ in range(self.sim_times):
            training_instance.reset()
            while True:
                if not training_instance.update():
                    break

            # snake has collided with the wall
            scores.append(training_instance.get_fitness())
        return statistics.mean(scores)

default_sim = 1

default_food_pos = None

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    grid_width, grid_height = 13, 13
    scores = []
    training_instance = TrainingInstance(grid_width, grid_height, net, default_food_pos)

    for _ in range(default_sim):
        training_instance.reset()
        while True:
            if not training_instance.update():
                break

        # snake has collided with the wall
        scores.append(training_instance.get_fitness())
    genome.food_seq = training_instance.food_seq
    return statistics.mean(scores)

gen_count = 1

def eval_genomes(genomes, config):
    nets = [neat.nn.FeedForwardNetwork.create(genome, config) for _, genome in genomes]

    global default_food_pos, gen_count
    default_food_pos = random_food_positions(50, 13, 13)

    playground = TrainingPlayground(13, 13, 1, default_food_pos)
    playground.train(list(map(lambda genome: genome[1], genomes)), nets)

    best_genome_id, best_genome = max(genomes, key=lambda x: x[1].fitness)

    # Write the best fitness to a file
    with open('best_fitness.csv', 'a+') as file:
        file.write(f'{gen_count}, {best_genome.fitness}, {playground.max_score}\n')

    gen_count += 1
