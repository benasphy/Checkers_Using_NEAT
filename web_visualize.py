from flask import Flask, render_template_string, request, jsonify
import numpy as np
from checkers.game import CheckersGame
from ai.agent import NEATAgent, ValueNEATAgent
from ai.random_agent import RandomAgent
from ai.mcts import MCTSAgent
import neat
import os
import pickle

app = Flask(__name__)

# HTML template for board rendering
HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Checkers Web Visualization</title>
  <style>
    body { font-family: sans-serif; }
    .board { display: grid; grid-template: repeat(8, 40px) / repeat(8, 40px); margin: 20px auto; }
    .cell { width: 40px; height: 40px; box-sizing: border-box; display: flex; align-items: center; justify-content: center; cursor: pointer; }
    .light { background: #e8ebef; }
    .dark { background: #7d8796; }
    .r { background: #e74c3c; border-radius: 50%; width: 32px; height: 32px; }
    .b { background: #222; border-radius: 50%; width: 32px; height: 32px; }
    .R { background: #e74c3c; border: 3px solid gold; border-radius: 50%; width: 32px; height: 32px; }
    .B { background: #222; border: 3px solid gold; border-radius: 50%; width: 32px; height: 32px; }
    .selected { outline: 2px solid #27ae60; }
  </style>
  <script>
    let selected = null;
    function selectCell(row, col) {
      // Deselect previous
      if (selected) {
        document.getElementById(selected).classList.remove('selected');
      }
      selected = row + '-' + col;
      document.getElementById(selected).classList.add('selected');
      document.getElementById('from_row').value = row;
      document.getElementById('from_col').value = col;
      // Enable only empty, dark cells as destination
      Array.from(document.getElementsByClassName('cell')).forEach(cell => {
        cell.classList.remove('can-select-dest');
      });
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          let cell = document.getElementById(`${r}-${c}`);
          if (cell && cell.dataset.empty === "1" && (r+c)%2 === 1) {
            cell.classList.add('can-select-dest');
            cell.onclick = function() { setToCell(r, c); };
          }
        }
      }
    }
    function setToCell(row, col) {
      if (!selected) return; // Must select a piece first
      document.getElementById('to_row').value = row;
      document.getElementById('to_col').value = col;
      // Reset selection before submitting
      document.getElementById(selected).classList.remove('selected');
      selected = null;
      Array.from(document.getElementsByClassName('cell')).forEach(cell => {
        cell.classList.remove('can-select-dest');
        cell.onclick = null;
      });
      document.getElementById('moveForm').submit();
    }
    // On page load, clear all hidden fields
    window.onload = function() {
      document.getElementById('from_row').value = '';
      document.getElementById('from_col').value = '';
      document.getElementById('to_row').value = '';
      document.getElementById('to_col').value = '';
    }
  </script>
</head>
<body>
  <h2>Checkers: Human vs NEAT</h2>
  <form method="post" style="margin-bottom: 10px;">
    <label>Agent Mode: </label>
    <select name="agent_mode" onchange="this.form.submit()">
      <option value="neat" {% if agent_mode == 'neat' %}selected{% endif %}>NEAT Only</option>
      <option value="mcts" {% if agent_mode == 'mcts' %}selected{% endif %}>MCTS + NEAT</option>
    </select>
    <label style="margin-left: 10px;">MCTS Simulations:</label>
    <select name="mcts_simulations" onchange="this.form.submit()">
      {% for n in [50, 100, 200, 400, 800, 1600] %}
        <option value="{{n}}" {% if mcts_simulations == n %}selected{% endif %}>{{n}}</option>
      {% endfor %}
    </select>
    <span style="margin-left: 10px; color: #888;">Current: {{ agent_mode|capitalize }}{% if agent_mode == 'mcts' %} ({{ mcts_simulations }} sims){% endif %}</span>
  </form>

  <div class="board">
    {% for row in range(8) %}
      {% for col in range(8) %}
        {% set cell_id = row|string + '-' + col|string %}
        <div class="cell {{ 'light' if (row+col)%2==0 else 'dark' }}" id="{{cell_id}}" data-empty="{{ 1 if board[row][col] == 0 else 0 }}"
          {% if human_turn and board[row][col] in [1,3] and from_row is none %}
            onclick="selectCell({{row}},{{col}})"
          {% endif %}
        >
          {% if board[row][col] == 1 %}<div class="r"></div>{% endif %}
          {% if board[row][col] == 2 %}<div class="b"></div>{% endif %}
          {% if board[row][col] == 3 %}<div class="R"></div>{% endif %}
          {% if board[row][col] == 4 %}<div class="B"></div>{% endif %}
        </div>
      {% endfor %}
    {% endfor %}
  </div>
  <p>{{ status }}</p>
  <form id="moveForm" method="post">
    <input type="hidden" name="from_row" id="from_row" value="{{ from_row if from_row is not none else '' }}">
    <input type="hidden" name="from_col" id="from_col" value="{{ from_col if from_col is not none else '' }}">
    <input type="hidden" name="to_row" id="to_row" value="">
    <input type="hidden" name="to_col" id="to_col" value="">
    {% if human_turn %}
      <div style="margin: 10px 0; color: #888;">Select your piece, then select a destination square. No need to press submit.</div>
    {% endif %}
    <button name="reset" value="1">Reset Game</button>
  </form>
</body>
</html>
'''

game = None
agent1 = None
agent2 = None
config = None
agent_mode = 'neat'  # 'neat' or 'mcts'
mcts_simulations = 200

def setup_agents():
    global agent1, agent2, config, agent_mode, mcts_simulations
    config_path = os.path.join(os.path.dirname(__file__), 'neat_config.txt')
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    # Load policy and value NEAT agents
    neat_agent = None
    value_agent = None
    try:
        with open('best_policy_genome.pkl', 'rb') as f:
            policy_genome = pickle.load(f)
        neat_agent = NEATAgent(policy_genome, config, player=2)
    except FileNotFoundError:
        try:
            with open('best_genome.pkl', 'rb') as f:
                policy_genome = pickle.load(f)
            neat_agent = NEATAgent(policy_genome, config, player=2)
        except FileNotFoundError:
            print("Warning: Could not load policy NEAT agent. Using RandomAgent instead.")
            neat_agent = RandomAgent(player=2)
    try:
        with open('best_value_genome.pkl', 'rb') as f:
            value_genome = pickle.load(f)
        value_agent = ValueNEATAgent(value_genome, config, player=2)
    except FileNotFoundError:
        value_agent = None
    agent1 = RandomAgent(player=1)
    if agent_mode == 'mcts':
        if value_agent is not None:
            agent2 = MCTSAgent(neat_agent, value_agent=value_agent, num_simulations=mcts_simulations, c_param=1.4)
        else:
            agent2 = MCTSAgent(neat_agent, num_simulations=mcts_simulations, c_param=1.4)
    else:
        agent2 = neat_agent

@app.route('/', methods=['GET', 'POST'])
def index():
    global game, agent1, agent2, agent_mode, mcts_simulations
    # Get agent mode and simulation count from query or form
    if request.method == 'POST':
        agent_mode = request.form.get('agent_mode', agent_mode)
        try:
            mcts_simulations = int(request.form.get('mcts_simulations', mcts_simulations))
        except (TypeError, ValueError):
            mcts_simulations = 100
    else:
        agent_mode = request.args.get('agent_mode', agent_mode)
        try:
            mcts_simulations = int(request.args.get('mcts_simulations', mcts_simulations))
        except (TypeError, ValueError):
            mcts_simulations = 100

    if game is None or request.form.get('reset'):
        game = CheckersGame()
        setup_agents()
    status = ""
    human_turn = (game.current_player == 1)
    from_row = request.form.get('from_row')
    from_col = request.form.get('from_col')
    to_row = request.form.get('to_row')
    to_col = request.form.get('to_col')
    move_made = False

    if request.method == 'POST' and request.form.get('reset'):
        game = CheckersGame()
        setup_agents()
        status = "Game reset. Human's turn."
        human_turn = True
        from_row = from_col = to_row = to_col = None
    elif request.method == 'POST' and human_turn:
        # Human move
        print(f"DEBUG: Human submitted move: from=({from_row},{from_col}) to=({to_row},{to_col})")
        if all(x not in (None, "") for x in [from_row, from_col, to_row, to_col]):
            try:
                move = (int(from_row), int(from_col), int(to_row), int(to_col))
                legal_moves = game.get_legal_moves()
                print(f"DEBUG: Legal moves: {legal_moves}")
                # Find the full move tuple to pass to game.make_move
                full_move = next((lm for lm in legal_moves if move == lm[:4]), None)
                if full_move:
                    game.make_move(full_move)
                    move_made = True
                    from_row = from_col = to_row = to_col = None
                else:
                    status = "Invalid move. Try again."
                    print(f"DEBUG: Invalid move attempted: {move}")
                    from_row = from_col = to_row = to_col = None
            except Exception as e:
                status = f"Error: {e}"
                print(f"DEBUG: Error processing move: {e}")
                to_row = to_col = None
    # Agent2 move if it's agent2's turn and game not over
    while not game.is_game_over() and game.current_player == 2:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break
        # For MCTS agent, pass the game object; for NEAT, pass board/legals
        if agent_mode == 'mcts' and isinstance(agent2, MCTSAgent):
            move = agent2.select_move(game)
        else:
            move = agent2.select_move(game.board.board, legal_moves)
        if move:
            game.make_move(move)
        else:
            break

    # Update status
    if game.is_game_over():
        winner = game.get_winner()
        if winner == 1:
            status = "Human wins!"
        elif winner == 2:
            status = f"{'MCTS+NEAT' if agent_mode == 'mcts' else 'NEAT'} Agent wins!"
        else:
            status = "Draw!"
    else:
        status = f"{'Human' if game.current_player == 1 else ('MCTS+NEAT' if agent_mode == 'mcts' else 'NEAT Agent')}'s turn"
    board = game.board.board.tolist()
    return render_template_string(
        HTML_TEMPLATE,
        board=board,
        status=status,
        human_turn=(game.current_player == 1 and not game.is_game_over()),
        from_row=from_row,
        from_col=from_col,
        agent_mode=agent_mode,
        mcts_simulations=mcts_simulations
    )

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
