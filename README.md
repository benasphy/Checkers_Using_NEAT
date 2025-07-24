# Checkers AI with NEAT

This project is a robust, feature-rich Checkers AI agent that learns to play Checkers using NEAT (NeuroEvolution of Augmenting Topologies). Inspired by AlphaGo, the agent evolves its neural network topology and weights through self-play and learns from its mistakes.

## Features
- Modular Checkers game engine (supports all rules, legal moves, win/draw detection)
- NEAT integration (using `neat-python`)
- Agent vs Agent and Agent vs Human play modes
- Training loop with logging and checkpointing
- Visualization (text-based and/or pygame for board display)
- Configurable NEAT parameters
- Easy extensibility for new features

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the AI:**
   ```bash
   python main.py train
   ```

3. **Play against the AI:**
   ```bash
   python main.py play
   ```

4. **Visualize games:**
   ```bash
   python main.py visualize
   ```

## Project Structure

- `checkers/` - Game logic and visualization
- `ai/` - AI agent, evaluation, and training code
- `neat_config.txt` - NEAT configuration
- `main.py` - CLI entry point

## Requirements
- Python 3.8+
- neat-python
- pygame
- numpy

## Extending
- You can add new reward functions, evaluation strategies, or swap out the neural evolution backend.

---
Inspired by AlphaGo and the power of neuroevolution!
