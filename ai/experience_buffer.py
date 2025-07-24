import os
import pickle
import random

class ExperienceReplayBuffer:
    def __init__(self, history_dir="analysis/game_histories"):
        self.history_dir = history_dir
        self.histories = []
        self.load_histories()

    def load_histories(self):
        self.histories = []
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
            return
        for fname in os.listdir(self.history_dir):
            if fname.endswith(".pkl"):
                with open(os.path.join(self.history_dir, fname), "rb") as f:
                    self.histories.append(pickle.load(f))

    def add_history(self, game_history, winner):
        os.makedirs(self.history_dir, exist_ok=True)
        fname = f"game_{int(random.random()*1e10)}.pkl"
        with open(os.path.join(self.history_dir, fname), "wb") as f:
            pickle.dump({'history': game_history, 'winner': winner}, f)
        self.histories.append({'history': game_history, 'winner': winner})

    def sample(self, batch_size=32):
        return random.sample(self.histories, min(batch_size, len(self.histories)))
