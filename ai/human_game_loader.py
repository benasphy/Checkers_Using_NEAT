import os
import pickle
import random

class HumanGameLoader:
    def __init__(self, human_game_dir="analysis/human_games"):
        self.human_game_dir = human_game_dir
        self.games = []
        self.load_games()

    def load_games(self):
        self.games = []
        if not os.path.exists(self.human_game_dir):
            return
        for fname in os.listdir(self.human_game_dir):
            if fname.endswith(".pkl"):
                with open(os.path.join(self.human_game_dir, fname), "rb") as f:
                    self.games.append(pickle.load(f))

    def sample(self, batch_size=32):
        return random.sample(self.games, min(batch_size, len(self.games)))
