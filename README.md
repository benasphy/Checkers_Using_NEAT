# Checkers AI with NEAT

This project is a robust, feature-rich Checkers AI agent that learns to play Checkers using NEAT (NeuroEvolution of Augmenting Topologies). Inspired by AlphaGo, the agent evolves its neural network topology and weights through self-play and learns from its mistakes.

---

## Features
- Modular Checkers game engine (supports all rules, legal moves, win/draw detection)
- NEAT integration (using `neat-python`)
- Multiple agent types: NEAT, MCTS+NEAT, Random, Value-based
- Agent vs Agent and Agent vs Human play modes
- Training loop with logging, checkpointing, and performance analysis
- Web-based play via Flask (with Render deployment support)
- Visualization (text-based, pygame, and web)
- Configurable NEAT parameters
- Easy extensibility for new features

---

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the AI
```bash
python main.py train
```

### 3. Play against the AI (CLI)
```bash
python main.py play
```

### 4. Visualize games (local, requires pygame)
```bash
python main.py visualize
```

### 5. Web App (Flask)
To run locally:
```bash
python web_visualize.py
```
Then open http://localhost:5000 in your browser.

### 6. Deploy to Render (Web)
- Push your code to GitHub
- Create a new Web Service on [Render](https://render.com/)
- Set the start command to:
  ```
  python web_visualize.py
  ```
- Your app will be available at a public URL

---

## Project Structure
- `checkers/` - Game logic and visualization
- `ai/` - AI agent, evaluation, and training code
- `main.py` - CLI entry point for training, playing, and visualizing
- `web_visualize.py` - Flask web app for browser-based play
- `neat_config.txt` - NEAT configuration file
- `requirements.txt` - Python dependencies
- `render.yaml` - Render deployment configuration
- `best_genome.pkl`, `best_policy_genome.pkl`, `best_value_genome.pkl` - Saved NEAT models
- `training_metrics.csv`, `fitness_log.csv`, `performance_log.pkl` - Training logs and metrics

---

## Web App Usage
- **Agent Mode:** Choose between NEAT Only or MCTS+NEAT (Monte Carlo Tree Search with NEAT policy/value guidance)
- **MCTS Simulations:** Set the number of simulations for MCTS agent
- **Reset Game:** Start a new game
- **How to Play:** Click a piece, then click a destination square. No need to press submit.

---

## AI Agents
- **NEATAgent:** Uses a neural network evolved by NEAT to select moves
- **MCTSAgent:** Uses Monte Carlo Tree Search, optionally guided by NEAT policy/value networks
- **RandomAgent:** Selects moves randomly (for baseline/testing)
- **ValueNEATAgent:** NEAT-evolved value network for board evaluation

---

## Training & Evaluation
- **Train NEAT:** `python main.py train`
- **Evaluate NEAT vs Random:** `python evaluate_neat_vs_random.py`
- **Visualize NEAT vs Random:** `python main.py viz_neat_vs_random`
- **Visualize Random vs Random:** `python main.py viz_random_vs_random`
- **Analysis & Plots:** Training metrics and performance logs are saved for later analysis (see `ai/game_analysis.py`)

---

## Extending
- Add new agent types in `ai/`
- Modify reward functions or evaluation strategies
- Change NEAT parameters in `neat_config.txt`
- Add new visualizations or web features

---

## Example Screenshot
![Web App Screenshot](<your-screenshot-url-here>)

---

## Troubleshooting
- **502 Bad Gateway on Render:** Ensure `web_visualize.py` uses the correct port:
  ```python
  import os
  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)
  ```
- **Missing .pkl files:** Train the agent first (`python main.py train`) or provide pre-trained models.
- **Pygame errors:** Some features require a local display (not available on Render).
- **Large files:** Do not commit files >100MB to GitHub. Use `.gitignore` for logs and checkpoints.

---

## Requirements
- Python 3.8+
- neat-python
- pygame
- numpy
- Flask

---

## Credits
Inspired by AlphaGo and the power of neuroevolution!

---

## License
MIT License (see [LICENSE](LICENSE) file)
