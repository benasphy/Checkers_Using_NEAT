import copy
import random
import numpy as np

class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        self.game = game  # Deep copy of the game state
        self.parent = parent
        self.move = move  # Move that led to this node
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = game.get_legal_moves(game.current_player)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        if not self.children:
            # No children to choose from; return None or raise a clear error
            raise ValueError("No children to select from in best_child. This usually means no moves are available.")
        choices_weights = [
            (child.value / (child.visits + 1e-6)) + c_param * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6))
            for child in self.children
        ]
        return self.children[int(np.argmax(choices_weights))]

    def expand(self):
        move = self.untried_moves.pop(random.randrange(len(self.untried_moves)))
        next_game = copy.deepcopy(self.game)
        next_game.make_move(move)
        child_node = MCTSNode(next_game, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def is_terminal(self):
        return self.game.is_game_over()

    def rollout_policy(self, legal_moves):
        return random.choice(legal_moves)

class MCTSAgent:
    def __init__(self, policy_agent, value_agent=None, num_simulations=200, c_param=1.4):
        self.policy_agent = policy_agent  # NEATAgent for moves
        self.value_agent = value_agent    # ValueNEATAgent for board eval
        self.num_simulations = num_simulations
        self.c_param = c_param

    def select_move(self, game):
        root = MCTSNode(copy.deepcopy(game))
        for _ in range(self.num_simulations):
            node = root
            # Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.c_param)
            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self.expand_with_policy(node)
            # Simulation
            reward = self.rollout(node.game)
            # Backpropagation
            self.backpropagate(node, reward)
        # Choose the move with the most visits
        if not root.children:
            return None  # No moves available
        best_child = max(root.children, key=lambda n: n.visits)
        return best_child.move

    def expand_with_policy(self, node):
        # Use policy_agent to bias which move to expand
        moves = node.untried_moves
        if hasattr(self.policy_agent, 'select_move') and len(moves) > 1:
            # Get policy outputs for all moves
            outputs = self.policy_agent.net.activate(np.array(getattr(node.game.board, 'board', node.game.board)).flatten() / 4.0)
            # Softmax over outputs for available moves
            move_scores = [outputs[idx] if idx < len(outputs) else -float('inf') for idx in range(len(moves))]
            exp_scores = np.exp(move_scores - np.max(move_scores))
            probs = exp_scores / np.sum(exp_scores)
            move_idx = np.random.choice(len(moves), p=probs)
        else:
            move_idx = random.randrange(len(moves))
        move = moves.pop(move_idx)
        next_game = copy.deepcopy(node.game)
        next_game.make_move(move)
        child_node = MCTSNode(next_game, parent=node, move=move)
        node.children.append(child_node)
        return child_node

    def rollout(self, game):
        rollout_game = copy.deepcopy(game)
        # If value_agent exists, use it for leaf eval
        if self.value_agent is not None:
            return self.value_agent.predict_value(rollout_game.board)
        # Fallback: play to terminal state
        while not rollout_game.is_game_over():
            legal_moves = rollout_game.get_legal_moves(rollout_game.current_player)
            if not legal_moves:
                break
            if rollout_game.current_player == self.policy_agent.player:
                move = self.policy_agent.select_move(rollout_game.board, legal_moves)
            else:
                move = random.choice(legal_moves)
            rollout_game.make_move(move)
        winner = rollout_game.get_winner()
        if winner == self.policy_agent.player:
            return 1.0
        elif winner == 0:
            return 0.0
        else:
            return -1.0

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            reward = -reward
            node = node.parent
