from ai.random_agent import RandomAgent

class GreedyAgent(RandomAgent):
    def select_move(self, board, legal_moves):
        # Prefer capturing moves, otherwise random
        capturing_moves = [m for m in legal_moves if len(m) > 4 and m[4]]
        if capturing_moves:
            return self.rng.choice(capturing_moves)
        return super().select_move(board, legal_moves)
