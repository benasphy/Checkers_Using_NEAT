import numpy as np

class Board:
    def __init__(self):
        self.reset()

    def reset(self):
        # 8x8 board: 0=empty, 1=player1, 2=player2, 3=player1 king, 4=player2 king
        self.board = np.zeros((8, 8), dtype=int)
        # Place pieces for both players
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row, col] = 2
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row, col] = 1

    def get_piece(self, row, col):
        return self.board[row, col]

    def set_piece(self, row, col, value):
        self.board[row, col] = value

    def copy(self):
        new_board = Board()
        new_board.board = self.board.copy()
        return new_board

    def get_legal_moves(self, player):
        # Returns a list of (from_row, from_col, to_row, to_col, [captures])
        # Standard American Checkers rules
        directions = {
            1: [(-1, -1), (-1, 1)],  # Player 1 moves up
            2: [(1, -1), (1, 1)],    # Player 2 moves down
            3: [(-1, -1), (-1, 1), (1, -1), (1, 1)],  # King
            4: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        }
        moves = []
        captures = []
        for row in range(8):
            for col in range(8):
                piece = self.get_piece(row, col)
                if (player == 1 and piece in [1, 3]) or (player == 2 and piece in [2, 4]):
                    dirs = directions[piece]
                    for dr, dc in dirs:
                        r, c = row + dr, col + dc
                        # Normal move
                        if 0 <= r < 8 and 0 <= c < 8 and self.get_piece(r, c) == 0:
                            moves.append((row, col, r, c, []))
                        # Capture
                        r2, c2 = row + 2 * dr, col + 2 * dc
                        if 0 <= r2 < 8 and 0 <= c2 < 8 and self.get_piece(r2, c2) == 0:
                            opponent = 2 if player == 1 else 1
                            opponent_k = 4 if player == 1 else 3
                            if self.get_piece(r, c) in [opponent, opponent_k]:
                                captures.append((row, col, r2, c2, [(r, c)]))
        return captures if captures else moves

    def is_game_over(self):
        # Game is over if either player has no pieces or no legal moves
        p1_pieces, p2_pieces = 0, 0
        for row in range(8):
            for col in range(8):
                piece = self.get_piece(row, col)
                if piece in [1, 3]:
                    p1_pieces += 1
                elif piece in [2, 4]:
                    p2_pieces += 1
        if p1_pieces == 0 or p2_pieces == 0:
            return True
        if not self.get_legal_moves(1) and not self.get_legal_moves(2):
            return True
        return False

    def __str__(self):
        # Simple text display
        symbols = {0: '.', 1: 'r', 2: 'b', 3: 'R', 4: 'B'}
        s = ''
        for row in range(8):
            for col in range(8):
                s += symbols[self.board[row, col]] + ' '
            s += '\n'
        return s
