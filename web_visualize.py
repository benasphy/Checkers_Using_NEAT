"""Flask web app: play against the evolved NEAT value network + alpha-beta agent.

The agent is the trained value genome (``best_value_genome.pkl``) evaluated
inside iterative-deepening alpha-beta search; the Strength control sets the
search depth. Falls back to a material-only searcher when no genome exists.
"""

import os
import pickle

from flask import Flask, render_template_string, request

from checkers.game import CheckersGame
from ai.ladder import make_agent, neat_spec

app = Flask(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "neat_value_config.txt")
GENOME_PATH = os.path.join(os.path.dirname(__file__), "best_value_genome.pkl")
MOVE_TIME_LIMIT = 5.0  # seconds per AI move in the web UI

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Checkers vs NEAT + Search</title>
  <style>
    body { font-family: sans-serif; text-align: center; }
    .board { display: grid; grid-template: repeat(8, 40px) / repeat(8, 40px); margin: 20px auto; width: 320px; }
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
      if (selected) document.getElementById(selected).classList.remove('selected');
      selected = row + '-' + col;
      document.getElementById(selected).classList.add('selected');
      document.getElementById('from_row').value = row;
      document.getElementById('from_col').value = col;
      for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
        let cell = document.getElementById(`${r}-${c}`);
        if (cell && cell.dataset.empty === "1" && (r + c) % 2 === 1) {
          cell.onclick = function () { setToCell(r, c); };
        }
      }
    }
    function setToCell(row, col) {
      if (!selected) return;
      document.getElementById('to_row').value = row;
      document.getElementById('to_col').value = col;
      document.getElementById('moveForm').submit();
    }
    window.onload = function () {
      ['from_row', 'from_col', 'to_row', 'to_col'].forEach(
        id => document.getElementById(id).value = '');
    }
  </script>
</head>
<body>
  <h2>Checkers: Human vs NEAT + Alpha-Beta</h2>
  <form method="post">
    <label>Strength (search depth): </label>
    <select name="depth" onchange="this.form.submit()">
      {% for d in [2, 4, 6, 8] %}
        <option value="{{d}}" {% if depth == d %}selected{% endif %}>depth {{d}}</option>
      {% endfor %}
    </select>
    <span style="margin-left: 10px; color: #888;">Agent: {{ agent_name }}</span>
  </form>

  <div class="board">
    {% for row in range(8) %}
      {% for col in range(8) %}
        {% set cell_id = row|string + '-' + col|string %}
        <div class="cell {{ 'light' if (row+col)%2==0 else 'dark' }}" id="{{cell_id}}"
             data-empty="{{ 1 if board[row][col] == 0 else 0 }}"
          {% if human_turn and board[row][col] in [1,3] %}
            onclick="selectCell({{row}},{{col}})"
          {% endif %}>
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
    <input type="hidden" name="from_row" id="from_row" value="">
    <input type="hidden" name="from_col" id="from_col" value="">
    <input type="hidden" name="to_row" id="to_row" value="">
    <input type="hidden" name="to_col" id="to_col" value="">
    {% if human_turn %}
      <div style="margin: 10px 0; color: #888;">
        Click a piece, then its destination. For a multi-jump, click the FINAL landing square.
        Captures are mandatory.
      </div>
    {% endif %}
    <button name="reset" value="1">Reset Game</button>
  </form>
</body>
</html>
'''

game = None
agent = None
agent_name = "?"
depth = 4


def setup_agent():
    global agent, agent_name
    if os.path.exists(GENOME_PATH):
        try:
            with open(GENOME_PATH, "rb") as f:
                genome = pickle.load(f)
            agent = make_agent(neat_spec(genome, CONFIG_PATH, depth))
            agent_name = f"NEAT value net + alpha-beta (d{depth})"
            return
        except Exception as e:  # incompatible/corrupt pickle
            print(f"Could not load genome ({e}); using material searcher.")
    agent = make_agent(("material", depth))
    agent_name = f"material-only alpha-beta (d{depth})"


@app.route("/", methods=["GET", "POST"])
def index():
    global game, depth
    status = ""

    if request.method == "POST" and request.form.get("depth"):
        try:
            new_depth = int(request.form["depth"])
            if new_depth != depth:
                depth = new_depth
                setup_agent()
        except ValueError:
            pass

    if game is None or (request.method == "POST" and request.form.get("reset")):
        game = CheckersGame()
        setup_agent()
        status = "New game. Your move."

    if request.method == "POST" and not request.form.get("reset") \
            and game.current_player == 1 and not game.is_game_over():
        vals = [request.form.get(k) for k in
                ("from_row", "from_col", "to_row", "to_col")]
        if all(v not in (None, "") for v in vals):
            fr_r, fr_c, to_r, to_c = map(int, vals)
            mv = game.find_engine_move((fr_r, fr_c), (to_r, to_c))
            if mv is not None:
                game.make_engine_move(mv)
            else:
                status = "Illegal move (captures are mandatory). Try again."

    # AI reply
    while not game.is_game_over() and game.current_player == 2:
        mv = agent.select(game, max_seconds=MOVE_TIME_LIMIT) \
            if hasattr(agent, "searcher") else agent.select(game)
        if mv is None:
            break
        game.make_engine_move(mv)

    if game.is_game_over():
        winner = game.get_winner()
        status = {1: "You win!", 2: "AI wins!",
                  0: f"Draw ({game.draw_reason()})."}[winner]
    elif not status:
        status = "Your move." if game.current_player == 1 else "AI thinking..."

    return render_template_string(
        HTML_TEMPLATE,
        board=game.position.to_array().tolist(),
        status=status,
        human_turn=(game.current_player == 1 and not game.is_game_over()),
        depth=depth,
        agent_name=agent_name,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
