from ai.agent import NEATAgent
from ai.random_agent import RandomAgent
from checkers.game import CheckersGame
import neat
import pickle

import concurrent.futures
import itertools

def _play_game(args):
    (policy_id, policy_genome_data, value_id, value_genome_data, config_policy, config_value, 
     opp_policy_data, opp_value_data, hall_of_fame, max_moves, mcts_simulations) = args
    from ai.agent import NEATAgent, ValueNEATAgent
    from ai.random_agent import RandomAgent
    from checkers.game import CheckersGame
    import numpy as np
    
    # Load genomes and create agents
    policy_genome = pickle.loads(policy_genome_data)
    value_genome = pickle.loads(value_genome_data)
    
    # Set up opponent agents
    if opp_policy_data is not None:
        opp_policy = pickle.loads(opp_policy_data)
        agent2 = NEATAgent(opp_policy, config_policy, player=2)
    else:
        # Use a mix of RandomAgent and GreedyAgent for more diverse opponents
        import random
        if random.random() < 0.5:
            from ai.random_agent import RandomAgent
            agent2 = RandomAgent(player=2)
        else:
            try:
                from ai.greedy_agent import GreedyAgent
                agent2 = GreedyAgent(player=2)
            except ImportError:
                from ai.random_agent import RandomAgent
                agent2 = RandomAgent(player=2)
        
    if opp_value_data is not None:
        opp_value = pickle.loads(opp_value_data)
        value2 = ValueNEATAgent(opp_value, config_value, player=2)
    else:
        value2 = None
        
    agent1 = NEATAgent(policy_genome, config_policy, player=1)
    value1 = ValueNEATAgent(value_genome, config_value, player=1)
    
    # Initialize metrics
    fitness1 = 0
    move_history = set()  # Track board states for repetition detection
    
    # Play two games (swapping sides)
    for swap in range(2):
        game = CheckersGame()
        done = False
        move_count = 0
        game.reset()
        repeated_positions = 0
        good_move_count = 0
        max_piece_advantage = 0
        
        while not done and move_count < max_moves:
            # Track board states for repetition detection
            board_state = str(game.board.board)
            if board_state in move_history:
                repeated_positions += 1
            else:
                move_history.add(board_state)
            
            # Get piece counts before move
            prev_pieces_2 = int((game.board.board == 2).sum() + (game.board.board == 4).sum())
            prev_pieces_1 = int((game.board.board == 1).sum() + (game.board.board == 3).sum())
            
            # Get and make move
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
                
            current_player = game.current_player
            if current_player == 1:
                move = agent1.select_move(game.board.board, legal_moves)
            else:
                move = agent2.select_move(game.board.board, legal_moves)
                
            if move:
                # Reward for making a non-losing move
                if current_player == 1:
                    good_move_count += 1
                    
                game.make_move(move)
                
            done = game.is_game_over()
            move_count += 1
            
            # Calculate piece advantage
            curr_pieces_2 = int((game.board.board == 2).sum() + (game.board.board == 4).sum())
            curr_pieces_1 = int((game.board.board == 1).sum() + (game.board.board == 3).sum())
            piece_advantage = curr_pieces_1 - curr_pieces_2
            max_piece_advantage = max(max_piece_advantage, piece_advantage)
            
            # Reward for capturing pieces
            if current_player == 1:
                if curr_pieces_2 < prev_pieces_2:
                    fitness1 += 0.2  # Increased reward for captures
                if curr_pieces_1 < prev_pieces_1:
                    fitness1 -= 0.3  # Penalty for losing pieces
        
        # Game over rewards
        my_pieces = int((game.board.board == 1).sum() + (game.board.board == 3).sum())
        opp_pieces = int((game.board.board == 2).sum() + (game.board.board == 4).sum())
        piece_advantage = my_pieces - opp_pieces
        
        winner = game.get_winner()
        if winner == 1:
            # Reward based on margin of victory and game length
            margin_bonus = 0.1 * piece_advantage
            speed_bonus = 0.05 * (max_moves - move_count)  # Faster wins get a small bonus
            fitness1 += 10 + margin_bonus + speed_bonus
        elif winner == 2:
            fitness1 -= 5
        else:  # Draw
            # Small penalty for draws, but less than losing
            fitness1 -= 2
            
        # Penalize repeated positions (discourage draw by repetition)
        fitness1 -= 0.5 * repeated_positions
        
        # Reward for good moves
        fitness1 += 0.05 * good_move_count
        
        # Reward for maintaining piece advantage
        fitness1 += 0.1 * max_piece_advantage
        
        # Swap sides for next game
        agent1, agent2 = agent2, agent1
        value1, value2 = value2, value1
        
    return (policy_id, value_id, max(0, fitness1))  # Ensure non-negative fitness

def evaluate_selfplay(policy_population, value_population, config_policy, config_value, hall_of_fame, games_per_genome=3, mcts_simulations=50, max_moves=100, executor=None):
    import pickle
    # Reset fitness
    for genome in policy_population.values():
        genome.fitness = 0
    for genome in value_population.values():
        genome.fitness = 0
    # Prepare all matchups
    tasks = []
    for policy_id, policy_genome in policy_population.items():
        for value_id, value_genome in value_population.items():
            opponents = hall_of_fame[:] if hall_of_fame else [(None, None)]
            for opp_policy, opp_value in opponents:
                for _ in range(games_per_genome):
                    tasks.append((policy_id, pickle.dumps(policy_genome), value_id, pickle.dumps(value_genome), config_policy, config_value, pickle.dumps(opp_policy) if opp_policy else None, pickle.dumps(opp_value) if opp_value else None, hall_of_fame, max_moves, mcts_simulations))
    # Parallel evaluation
    if executor is not None:
        results = list(executor.map(_play_game, tasks))
    else:
        results = list(map(_play_game, tasks))
    # Aggregate fitness
    for policy_id, value_id, fitness1 in results:
        policy_population[policy_id].fitness += fitness1
        value_population[value_id].fitness += fitness1
    # Ensure all genomes have numeric fitness
    for genome in policy_population.values():
        if genome.fitness is None:
            genome.fitness = 0.0
    for genome in value_population.values():
        if genome.fitness is None:
            genome.fitness = 0.0

    from ai.agent import NEATAgent, ValueNEATAgent
    from ai.random_agent import RandomAgent
    from checkers.game import CheckersGame
    import numpy as np
    # Reset fitness
    for genome in policy_population.values():
        genome.fitness = 0
    for genome in value_population.values():
        genome.fitness = 0
    # Evaluate each policy-value pair
    for policy_idx, (policy_id, policy_genome) in enumerate(policy_population.items()):
        for value_idx, (value_id, value_genome) in enumerate(value_population.items()):
            print(f"Evaluating Policy {policy_idx+1}/{len(policy_population)} vs Value {value_idx+1}/{len(value_population)}")
            agent1 = NEATAgent(policy_genome, config_policy, player=1)
            value1 = ValueNEATAgent(value_genome, config_value, player=1)
            # Play against a sample from hall of fame or random
            opponents = hall_of_fame[:] if hall_of_fame else [(None, None)]
            for opp_policy, opp_value in opponents:
                if opp_policy is not None and opp_value is not None:
                    agent2 = NEATAgent(opp_policy, config_policy, player=2)
                    value2 = ValueNEATAgent(opp_value, config_value, player=2)
                else:
                    agent2 = RandomAgent(player=2)
                    value2 = None
                # Play two games (swap sides)
                for swap in range(2):
                    print(f"  Game swap {swap+1}/2 vs opponent {opponents.index((opp_policy, opp_value))+1}/{len(opponents)}")
                    game = CheckersGame()
                    fitness1 = 0
                    fitness2 = 0
                    done = False
                    move_count = 0
                    game.reset()
                    while not done and move_count < max_moves:
                        prev_pieces_2 = int((game.board.board == 2).sum() + (game.board.board == 4).sum())
                        prev_pieces_1 = int((game.board.board == 1).sum() + (game.board.board == 3).sum())
                        legal_moves = game.get_legal_moves()
                        if not legal_moves:
                            break
                        if game.current_player == 1:
                            move = agent1.select_move(game.board.board, legal_moves)
                        else:
                            move = agent2.select_move(game.board.board, legal_moves)
                        if move:
                            game.make_move(move)
                        done = game.is_game_over()
                        move_count += 1
                        # Reward for surviving a move
                        fitness1 += 0.005 if game.current_player == 2 else 0
                        fitness2 += 0.005 if game.current_player == 1 else 0
                        # Reward for capturing a piece
                        curr_pieces_2 = int((game.board.board == 2).sum() + (game.board.board == 4).sum())
                        curr_pieces_1 = int((game.board.board == 1).sum() + (game.board.board == 3).sum())
                        if curr_pieces_2 < prev_pieces_2:
                            fitness1 += 0.1
                        if curr_pieces_1 < prev_pieces_1:
                            fitness2 += 0.1
                    # Piece advantage at end
                    my_pieces = int((game.board.board == 1).sum() + (game.board.board == 3).sum())
                    opp_pieces = int((game.board.board == 2).sum() + (game.board.board == 4).sum())
                    fitness1 += 0.1 * (my_pieces - opp_pieces)
                    winner = game.get_winner()
                    if winner == 1:
                        fitness1 += 1
                    elif winner == 2:
                        fitness1 -= 1
                    # Assign fitness
                    policy_genome.fitness += fitness1
                    value_genome.fitness += fitness1
                    # Swap sides for next game
                    agent1, agent2 = agent2, agent1
                    value1, value2 = value2, value1

    # Ensure all genomes have numeric fitness
    for genome in policy_population.values():
        if genome.fitness is None:
            genome.fitness = 0.0
    for genome in value_population.values():
        if genome.fitness is None:
            genome.fitness = 0.0
