import neat
from snaketrain import eval_genome, eval_genomes, default_food_pos
import snaketrain
import multiprocessing
from game import SnakeGame
import argparse

if __name__ == "__main__":
    # command line argument
    parser = argparse.ArgumentParser(
                    prog='SnakeAI',
                    description='bla bla bla',
                    epilog='bla bla bla')

    parser.add_argument('-p', '--play', action='store_true')  
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--gen')
    parser.add_argument('--replay', action='store_true')

    args = parser.parse_args()

    config_file = args.config or "nn-config"
    checkpoint = args.checkpoint or None
    gen_count = int(args.gen) or 1
    replay = args.replay or None

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    if checkpoint:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    if replay:
        snaketrain.default_food_pos = snaketrain.random_food_positions(50, 13, 13)


    if args.parallel:
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
        winner = p.run(pe.evaluate, gen_count)
    else:
        winner = p.run(eval_genomes, gen_count)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    if args.play:
        print("...")
        game = SnakeGame(winner_net, snaketrain.default_food_pos)
        game.run()
