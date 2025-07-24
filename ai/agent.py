import neat
import numpy as np

class NEATAgent:
    def __init__(self, genome, config, player=1):
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.player = player

    def select_move(self, board, legal_moves):
        # Support both Board object and np.ndarray
        board_arr = getattr(board, 'board', board)
        input_board = np.array(board_arr).flatten() / 4.0  # Normalize values
        outputs = self.net.activate(input_board)
        move_scores = []
        for idx, move in enumerate(legal_moves):
            move_score = outputs[idx] if idx < len(outputs) else -float('inf')
            move_scores.append(move_score)
        best_idx = int(np.argmax(move_scores))
        return legal_moves[best_idx]

    def learn_from_experience(self, board_before, move, winner):
        # Placeholder: In NEAT, direct online learning is not standard.
        # You can use this to collect data for supervised or imitation learning.
        # For now, just pass. Implement custom logic if you switch to a trainable NN.
        pass


class ValueNEATAgent:
    def __init__(self, genome, config, player=2):
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.player = player

    def predict_value(self, board):
        board_arr = getattr(board, 'board', board)
        input_board = np.array(board_arr).flatten() / 4.0  # Normalize values
        # Output is a single value between -1 and 1
        value = self.net.activate(input_board)[0]
        return value

