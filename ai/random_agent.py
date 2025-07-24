import random

class RandomAgent:
    def __init__(self, player=2):
        self.player = player
        self.rng = random.Random()

    def select_move(self, board, legal_moves):
        return random.choice(legal_moves) if legal_moves else None
