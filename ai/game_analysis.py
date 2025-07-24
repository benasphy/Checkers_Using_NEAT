"""
Game Analysis Tools for Checkers AI
- Game visualization and replay
- Move analysis and blunder detection
- Performance tracking
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from checkers.game import CheckersGame
from checkers.visualize import Visualizer

class GameAnalyzer:
    def __init__(self, policy_agent=None, value_agent=None):
        self.policy_agent = policy_agent
        self.value_agent = value_agent
        self.performance_log = []
        self.move_history = []
        
    def record_game(self, game_history):
        """Record a game's moves and states for later analysis."""
        self.move_history.append({
            'timestamp': datetime.now(),
            'moves': game_history,
            'result': game_history[-1]['result'] if game_history else None
        })
        
    def visualize_game(self, game_history, delay=0.5):
        """Replay and visualize a game move by move."""
        if not game_history:
            print("No game history to visualize")
            return
            
        game = CheckersGame()
        vis = Visualizer(game.board)
        
        print("Starting game visualization...")
        print(f"Final result: {game_history[-1]['result']}")
        
        for move_data in game_history:
            vis.draw_board()
            vis.show()
            
            if 'move' in move_data and move_data['move']:
                print(f"\nMove {move_data['move_number']}: {move_data['move']}")
                print(f"Player: {move_data['player']}")
                if 'evaluation' in move_data:
                    print(f"Evaluation: {move_data['evaluation']:.3f}")
                if 'is_blunder' in move_data and move_data['is_blunder']:
                    print("⚠️  BLUNDER DETECTED!")
                
                game.make_move(move_data['move'])
                
            input("Press Enter to continue...")
            
    def analyze_moves(self, game_history):
        """Analyze moves for blunders and inaccuracies."""
        if not game_history or not self.value_agent:
            return []
            
        analysis = []
        for i, move_data in enumerate(game_history):
            if 'move' not in move_data:
                continue
                
            board_before = move_data['board_before']
            board_after = move_data['board_after']
            
            # Get evaluation before and after move
            eval_before = self.evaluate_position(board_before)
            eval_after = self.evaluate_position(board_after)
            
            # Simple blunder detection (large evaluation swing)
            eval_swing = eval_before - eval_after
            is_blunder = (eval_swing > 1.0)  # Threshold for blunder
            
            analysis.append({
                'move_number': i + 1,
                'move': move_data['move'],
                'player': move_data['player'],
                'evaluation_before': eval_before,
                'evaluation_after': eval_after,
                'eval_swing': eval_swing,
                'is_blunder': is_blunder
            })
            
        return analysis
        
    def evaluate_position(self, board):
        """Evaluate a board position using the value network."""
        if not self.value_agent:
            return 0.0
            
        # Normalize board for neural network
        board_array = np.array(board).flatten() / 4.0
        evaluation = self.value_agent.net.activate(board_array)[0]
        return evaluation
        
    def track_performance(self, agent_name, result, metrics=None):
        """Track agent performance over time."""
        entry = {
            'timestamp': datetime.now(),
            'agent': agent_name,
            'result': result,
            'metrics': metrics or {}
        }
        self.performance_log.append(entry)
        self.save_performance_log()
        
    def save_performance_log(self, filename='performance_log.pkl'):
        """Save performance data to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.performance_log, f)
            
    def load_performance_log(self, filename='performance_log.pkl'):
        """Load performance data from a file."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.performance_log = pickle.load(f)
                
    def plot_performance(self, agent_name=None, window_size=10):
        """Plot performance metrics over time."""
        if not self.performance_log:
            print("No performance data available")
            return
            
        # Filter by agent if specified
        logs = [log for log in self.performance_log 
               if agent_name is None or log['agent'] == agent_name]
               
        if not logs:
            print(f"No data found for agent: {agent_name}")
            return
            
        # Calculate moving average of win rate
        results = [1 if log['result'] == 'win' else 0 for log in logs]
        moving_avg = np.convolve(results, np.ones(window_size)/window_size, mode='valid')
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(moving_avg)), moving_avg, label='Win Rate (Moving Avg)')
        plt.axhline(y=0.5, color='r', linestyle='--', label='50% Win Rate')
        plt.title(f'Agent Performance Over Time ({agent_name or "All Agents"})')
        plt.xlabel('Game Number')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True)
        plt.show()

def record_training_metrics(generation, policy_fitness, value_fitness, filename='training_metrics.csv'):
    """Record training metrics to a CSV file."""
    header = not os.path.exists(filename)
    with open(filename, 'a') as f:
        if header:
            f.write('timestamp,generation,policy_fitness,value_fitness\n')
        timestamp = datetime.now().isoformat()
        f.write(f'{timestamp},{generation},{policy_fitness},{value_fitness}\n')

def plot_training_metrics(filename='training_metrics.csv'):
    """Plot training metrics from a CSV file."""
    import pandas as pd
    
    if not os.path.exists(filename):
        print(f"No training metrics found at {filename}")
        return
        
    df = pd.read_csv(filename)
    if df.empty:
        print("No data to plot")
        return
        
    plt.figure(figsize=(12, 6))
    plt.plot(df['generation'], df['policy_fitness'], label='Policy Fitness')
    plt.plot(df['generation'], df['value_fitness'], label='Value Fitness')
    plt.title('Training Progress')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Example usage
    analyzer = GameAnalyzer()
    
    # After a game:
    # analyzer.record_game(game_history)
    # analyzer.visualize_game(game_history)
    # analysis = analyzer.analyze_moves(game_history)
    # analyzer.track_performance("my_agent", "win", {"moves": len(game_history)})
    # analyzer.plot_performance("my_agent")
    pass
