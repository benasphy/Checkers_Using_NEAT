import time
from checkers.game import CheckersGame
from checkers.visualize import Visualizer
from .agent import NEATAgent
from .random_agent import RandomAgent
import neat


def play_match(agent1, agent2, delay=0.5, save_history=False, history_dir="analysis/game_histories"):
    import os
    import pickle
    game = CheckersGame()
    vis = Visualizer(game.board)
    done = False
    game_history = []
    while not done:
        vis.draw_board()
        vis.show()
        time.sleep(delay)
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break
        board_before = game.board.board.copy()
        if game.current_player == agent1.player:
            move = agent1.select_move(game.board.board, legal_moves)
        else:
            move = agent2.select_move(game.board.board, legal_moves)
        if move:
            game.make_move(move)
            if save_history:
                game_history.append({
                    'player': 1 if game.current_player == 2 else 2,
                    'move': move,
                    'board_before': board_before,
                    'board_after': game.board.board.copy()
                })
        done = game.is_game_over()
    vis.draw_board()
    vis.show()
    winner = game.get_winner()
    if save_history:
        os.makedirs(history_dir, exist_ok=True)
        fname = f"game_{int(time.time())}.pkl"
        with open(f"{history_dir}/{fname}", "wb") as f:
            pickle.dump({'history': game_history, 'winner': winner}, f)
    return winner


def visualize_neat_vs_random(genome, config, delay=0.5):
    neat_agent = NEATAgent(genome, config, player=1)
    random_agent = RandomAgent(player=2)
    winner = play_match(neat_agent, random_agent, delay)
    print(f"Winner: {'NEAT Agent' if winner == 1 else 'Random Agent' if winner == 2 else 'Draw'}")


def visualize_random_vs_random(delay=0.5):
    agent1 = RandomAgent(player=1)
    agent2 = RandomAgent(player=2)
    winner = play_match(agent1, agent2, delay)
    print(f"Winner: {'Player 1' if winner == 1 else 'Player 2' if winner == 2 else 'Draw'}")
