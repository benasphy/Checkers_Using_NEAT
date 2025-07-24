import pygame
from .board import Board

CELL_SIZE = 60
BOARD_SIZE = 8

COLORS = {
    'light': (232, 235, 239),
    'dark': (125, 135, 150),
    'red': (255, 80, 80),
    'black': (70, 70, 70),
    'red_king': (255, 0, 0),
    'black_king': (0, 0, 0),
}

PIECE_RADIUS = CELL_SIZE // 2 - 8

class Visualizer:
    def __init__(self, board: Board):
        pygame.init()
        self.screen = pygame.display.set_mode((CELL_SIZE * BOARD_SIZE, CELL_SIZE * BOARD_SIZE))
        pygame.display.set_caption('Checkers')
        self.board = board

    def draw_board(self):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = COLORS['light'] if (row + col) % 2 == 0 else COLORS['dark']
                pygame.draw.rect(
                    self.screen, color,
                    (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )
                piece = self.board.get_piece(row, col)
                if piece == 1:
                    pygame.draw.circle(self.screen, COLORS['red'],
                        (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), PIECE_RADIUS)
                elif piece == 2:
                    pygame.draw.circle(self.screen, COLORS['black'],
                        (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), PIECE_RADIUS)
                elif piece == 3:
                    pygame.draw.circle(self.screen, COLORS['red_king'],
                        (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), PIECE_RADIUS)
                    pygame.draw.circle(self.screen, (255, 255, 0),
                        (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), PIECE_RADIUS // 2)
                elif piece == 4:
                    pygame.draw.circle(self.screen, COLORS['black_king'],
                        (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), PIECE_RADIUS)
                    pygame.draw.circle(self.screen, (255, 255, 0),
                        (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), PIECE_RADIUS // 2)

    def show(self):
        self.draw_board()
        pygame.display.flip()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()
