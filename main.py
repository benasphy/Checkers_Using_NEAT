import argparse
from ai.train import run_neat
from checkers.game import CheckersGame
from checkers.visualize import Visualizer
import os

def cli():
    parser = argparse.ArgumentParser(description='Checkers AI with NEAT')
    parser.add_argument('mode', choices=['train', 'play', 'visualize', 'viz_neat_vs_random', 'viz_random_vs_random'], help='Mode: train, play, visualize, viz_neat_vs_random, viz_random_vs_random')
    args = parser.parse_args()

    if args.mode == 'train':
        config_path = os.path.join(os.path.dirname(__file__), 'neat_config.txt')
        run_neat(config_path)
    elif args.mode == 'play':
        # Placeholder: Human vs AI play
        game = CheckersGame()
        print('Human vs AI not implemented yet.')
        print(game.board)
    elif args.mode == 'visualize':
        game = CheckersGame()
        vis = Visualizer(game.board)
        vis.show()
    elif args.mode == 'viz_neat_vs_random':
        import neat
        from ai.visualize_match import visualize_neat_vs_random
        config_path = os.path.join(os.path.dirname(__file__), 'neat_config.txt')
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        # Load or evolve a genome (for demo, evolve one quickly)
        from ai.train import run_neat
        genome = run_neat(config_path)
        visualize_neat_vs_random(genome, config)
    elif args.mode == 'viz_random_vs_random':
        from ai.visualize_match import visualize_random_vs_random
        visualize_random_vs_random()

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    # Use host="0.0.0.0" for Render compatibility
    cli()
