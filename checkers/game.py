from .board import Board

class CheckersGame:
    def __init__(self):
        self.board = Board()
        self.current_player = 1
        self.history = []

    def reset(self):
        self.board.reset()
        self.current_player = 1
        self.history = []

    def get_legal_moves(self, player=None):
        # Placeholder: implement full move generation logic
        # Should return list of moves [(from_row, from_col, to_row, to_col, [captures])]
        return self.board.get_legal_moves(player or self.current_player)

    def make_move(self, move):
        # move: (from_row, from_col, to_row, to_col, [captures])
        from_row, from_col, to_row, to_col, captures = move
        piece = self.board.get_piece(from_row, from_col)
        self.board.set_piece(from_row, from_col, 0)
        self.board.set_piece(to_row, to_col, piece)
        # Remove captured pieces
        for r, c in captures:
            self.board.set_piece(r, c, 0)
        # Kinging
        if piece == 1 and to_row == 0:
            self.board.set_piece(to_row, to_col, 3)
        elif piece == 2 and to_row == 7:
            self.board.set_piece(to_row, to_col, 4)
        # Switch player if no multi-capture available
        if captures:
            # Check if another capture is possible from new position
            new_moves = self.board.get_legal_moves(self.current_player)
            further_capture = False
            for nm in new_moves:
                if nm[0] == to_row and nm[1] == to_col and nm[4]:
                    further_capture = True
                    break
            if further_capture:
                # Multi-capture: same player moves again
                pass
            else:
                self.current_player = 2 if self.current_player == 1 else 1
        else:
            self.current_player = 2 if self.current_player == 1 else 1
        self.history.append(move)

    def is_game_over(self):
        # Placeholder: implement end condition
        return self.board.is_game_over()

    def get_winner(self):
        # Returns 1 if player 1 wins, 2 if player 2 wins, 0 for draw, None if not over
        p1_pieces, p2_pieces = 0, 0
        for row in range(8):
            for col in range(8):
                piece = self.board.get_piece(row, col)
                if piece in [1, 3]:
                    p1_pieces += 1
                elif piece in [2, 4]:
                    p2_pieces += 1
        if p1_pieces == 0 and p2_pieces == 0:
            return 0
        if p1_pieces == 0:
            return 2
        if p2_pieces == 0:
            return 1
        # Check for legal moves
        p1_moves = self.board.get_legal_moves(1)
        p2_moves = self.board.get_legal_moves(2)
        if not p1_moves and not p2_moves:
            return 0
        if not p1_moves:
            return 2
        if not p2_moves:
            return 1
        return None

    def get_state(self):
        # Returns a flattened board and current player
        return self.board.board.flatten(), self.current_player
