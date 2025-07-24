import pickle
from checkers.game import CheckersGame
from ai.agent import NEATAgent
from ai.random_agent import RandomAgent
import neat

config_path = 'neat_config.txt'
with open('best_genome.pkl', 'rb') as f:
    genome = pickle.load(f)
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

agent1 = NEATAgent(genome, config, player=1)
agent2 = RandomAgent(player=2)

results = {'neat': 0, 'random': 0, 'draw': 0}
N = 100  # Number of games
for i in range(N):
    if i % 10 == 0:
        print(f"Playing game {i+1}/{N}...")
    game = CheckersGame()
    move_count = 0
    max_moves = 100
    while not game.is_game_over() and move_count < max_moves:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break
        if game.current_player == 1:
            move = agent1.select_move(game.board.board, legal_moves)
        else:
            move = agent2.select_move(game.board.board, legal_moves)
        if move:
            game.make_move(move)
        move_count += 1
    if move_count >= max_moves:
        print(f"Max moves reached in game {i+1}, treating as draw.")
    winner = game.get_winner()
    if winner == 1:
        results['neat'] += 1
    elif winner == 2:
        results['random'] += 1
    else:
        results['draw'] += 1

print(f"Results after {N} games:")
print(f"NEAT wins: {results['neat']}")
print(f"RandomAgent wins: {results['random']}")
print(f"Draws: {results['draw']}")
