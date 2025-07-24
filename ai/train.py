import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import neat
import pickle
import os
import numpy as np
from datetime import datetime
from concurrent import futures
import concurrent.futures
from ai.evaluate import evaluate_selfplay
from ai.agent import NEATAgent, ValueNEATAgent
from ai.game_analysis import GameAnalyzer, record_training_metrics, plot_training_metrics
from ai.experience_buffer import ExperienceReplayBuffer
from ai.human_game_loader import HumanGameLoader
from checkers.game import CheckersGame

HALL_OF_FAME_SIZE = 5

def run_neat_dual(config_file, generations=50, enable_analysis=True):
    # Policy network population
    config_policy = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    # Value network population (separate config for flexibility)
    config_value = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    pop_policy = neat.Population(config_policy)
    pop_value = neat.Population(config_value)
    pop_policy.add_reporter(neat.StdOutReporter(True))
    pop_value.add_reporter(neat.StdOutReporter(True))
    stats_policy = neat.StatisticsReporter()
    stats_value = neat.StatisticsReporter()
    pop_policy.add_reporter(stats_policy)
    pop_value.add_reporter(stats_value)

    # Hall of fame: list of (policy_genome, value_genome)
    hall_of_fame = []
    best_pair = (None, None)
    best_fitness = float('-inf')

    # Initialize game analyzer if enabled
    analyzer = None
    if enable_analysis:
        analyzer = GameAnalyzer()
        print("Game analysis enabled. Tracking training progress...")

    # Track best genomes
    best_fitness = -float('inf')
    best_pair = None

    # Training loop
    for generation in range(generations):
        print(f"\n--- Generation {generation + 1}/{generations} ---")
        start_time = datetime.now()

        # Evaluate all pairs by self-play against hall of fame
        # Parallelized evaluation
        with concurrent.futures.ProcessPoolExecutor() as executor:
            evaluate_selfplay(pop_policy.population, pop_value.population, config_policy, config_value, hall_of_fame, games_per_genome=3, mcts_simulations=50, executor=executor)

        # Get best genomes
        best_policy = max(pop_policy.population.values(), key=lambda x: x.fitness)
        best_value = max(pop_value.population.values(), key=lambda x: x.fitness)

        # Record training metrics
        if enable_analysis:
            record_training_metrics(
                generation=generation + 1,
                policy_fitness=best_policy.fitness,
                value_fitness=best_value.fitness
            )

            # Sample a game for analysis
            if generation % 5 == 0:  # Every 5 generations
                sample_agent = NEATAgent(best_policy, config_policy, player=1)
                analyzer.policy_agent = sample_agent
                analyzer.value_agent = ValueNEATAgent(best_value, config_value, player=1)

                # Play a sample game
                game = CheckersGame()
                game_history = []
                move_count = 0

                while not game.is_game_over() and move_count < 100:  # Max 100 moves
                    legal_moves = game.get_legal_moves()
                    if not legal_moves:
                        break

                    # Record game state before move
                    board_before = np.copy(game.board.board)

                    # Make move
                    move = sample_agent.select_move(game.board.board, legal_moves)
                    game.make_move(move)

                    # Record move data
                    game_history.append({
                        'move_number': move_count + 1,
                        'player': 1 if game.current_player == 2 else 2,  # Player who made the move
                        'move': move,
                        'board_before': board_before,
                        'board_after': np.copy(game.board.board),
                        'result': None  # Will be updated after game ends
                    })

                    move_count += 1

                # Record game result
                winner = game.get_winner()
                result = 'win' if winner == 1 else 'loss' if winner == 2 else 'draw'
                for move_data in game_history:
                    move_data['result'] = result

                analyzer.record_game(game_history)
                analyzer.track_performance(
                    f"gen_{generation + 1}",
                    result,
                    {
                        'moves': move_count,
                        'policy_fitness': best_policy.fitness,
                        'value_fitness': best_value.fitness
                    }
                )

        # Advance both populations
        pop_policy.reporters.start_generation(generation)
        pop_value.reporters.start_generation(generation)
        pop_policy.population = pop_policy.reproduction.reproduce(pop_policy.config, pop_policy.species, pop_policy.config.pop_size, pop_policy.generation)
        pop_value.population = pop_value.reproduction.reproduce(pop_value.config, pop_value.species, pop_value.config.pop_size, pop_value.generation)
        pop_policy.species.speciate(pop_policy.config, pop_policy.population, generation)
        pop_value.species.speciate(pop_value.config, pop_value.population, generation)
        pop_policy.reporters.end_generation(pop_policy.config, pop_policy.population, pop_policy.species)
        pop_value.reporters.end_generation(pop_value.config, pop_value.population, pop_value.species)

        # Ensure all genomes have numeric fitness (again, in case NEAT added new genomes)
        for genome in pop_policy.population.values():
            if genome.fitness is None:
                genome.fitness = 0.0
        for genome in pop_value.population.values():
            if genome.fitness is None:
                genome.fitness = 0.0
        # Find best policy and value genomes
        best_policy = max(pop_policy.population.values(), key=lambda g: g.fitness)
        best_value = max(pop_value.population.values(), key=lambda g: g.fitness)
        # Save to hall of fame
        hall_of_fame.append((pickle.loads(pickle.dumps(best_policy)), pickle.loads(pickle.dumps(best_value))))
        hall_of_fame = sorted(hall_of_fame, key=lambda pair: pair[0].fitness + pair[1].fitness, reverse=True)[:HALL_OF_FAME_SIZE]
        # Save best overall
        if best_policy.fitness + best_value.fitness > best_fitness:
            best_fitness = best_policy.fitness + best_value.fitness
            best_pair = (pickle.loads(pickle.dumps(best_policy)), pickle.loads(pickle.dumps(best_value)))
        # Save checkpoints and show progress
        gen_time = (datetime.now() - start_time).total_seconds()
        print(f"Generation {generation + 1} completed in {gen_time:.1f}s")
        print(f"Best policy fitness: {best_policy.fitness:.3f}, "
              f"Best value fitness: {best_value.fitness:.3f}")
        
        # Save checkpoints
        with open('best_policy_genome.pkl', 'wb') as f:
            pickle.dump(best_pair[0], f)
        with open('best_value_genome.pkl', 'wb') as f:
            pickle.dump(best_pair[1], f)
        
        # Save generation checkpoint
        if (generation + 1) % 5 == 0:  # Every 5 generations
            checkpoint_name = f'checkpoint_gen_{generation + 1}'
            # pop_policy.save(os.path.join('checkpoints', f'{checkpoint_name}_policy'))
            # pop_value.save(os.path.join('checkpoints', f'{checkpoint_name}_value'))
            print(f"Saved checkpoint: {checkpoint_name}")
        
        # Experience Replay and Imitation Learning
        # (Disabled: user does not want to use it)
        # if generation % 5 == 0:
        #     exp_buffer = ExperienceReplayBuffer()
        #     human_loader = HumanGameLoader()
        #     retrain_from_experience(sample_agent, exp_buffer, epochs=1, batch_size=16)
        #     imitation_learning(sample_agent, human_loader, epochs=1, batch_size=16)
    
    # Final analysis
    if enable_analysis:
        print("\n--- Training Complete ---")
        print("Generating performance plots...")
        
        # Plot training metrics
        plot_training_metrics()
        
        # Plot performance over time
        analyzer.plot_performance()
        
        # Save analysis results
        analyzer.save_performance_log('training_analysis.pkl')
        print("Analysis results saved to 'training_analysis.pkl'")
    
    print("\nTraining complete. Best policy and value genomes saved.")
    return best_pair

def ensure_directory(directory):
    """Ensure a directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def retrain_from_experience(agent, buffer, epochs=1, batch_size=32):
    for _ in range(epochs):
        batch = buffer.sample(batch_size)
        for game in batch:
            for step in game['history']:
                agent.learn_from_experience(step['board_before'], step['move'], game['winner'])

def imitation_learning(agent, human_loader, epochs=1, batch_size=32):
    for _ in range(epochs):
        batch = human_loader.sample(batch_size)
        for game in batch:
            for step in game['history']:
                agent.learn_from_experience(step['board_before'], step['move'], game['winner'])

if __name__ == '__main__':
    # Setup directories
    local_dir = os.path.dirname(__file__)
    ensure_directory('checkpoints')
    ensure_directory('analysis')
    
    # Load config
    config_path = os.path.join(local_dir, '../neat_config.txt')
    
    # Start training with analysis enabled
    run_neat_dual(
        config_file=config_path,
        generations=50,
        enable_analysis=True
    )
