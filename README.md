# Cliff World: Q-Learning vs Expected Sarsa

This project implements and compares **Q-Learning** and **Expected Sarsa** reinforcement learning algorithms on the classic **Cliff Walking** environment. It is built using the [RL-Glue](http://rl-glue.ai/) interface and visualizes agent behavior, reward convergence, and policy safety.

## Overview

- Implemented two foundational RL algorithms:
  - **Q-Learning**: Off-policy, bootstrapping algorithm using max future rewards.
  - **Expected Sarsa**: On-policy algorithm using expected action values under the current policy.
- Evaluated both methods on the **Cliff World**, a deterministic gridworld where falling off the cliff incurs heavy penalties.
- Compared agent performance over multiple runs, step-sizes, and reward trends.

## Project Structure

```
├── agent.py                   # Base agent interface
├── q_learning_agent.py       # Q-Learning implementation
├── expected_sarsa_agent.py   # Expected Sarsa implementation
├── cliffworld_env.py         # Cliff World environment (provided or implemented separately)
├── run_experiments.ipynb     # Notebook for training, evaluation, and plotting
├── README.md                 # This file
 Results
```

Q-Learning learns the shortest path along the cliff but is prone to falling due to its off-policy nature.
Expected Sarsa follows safer routes and is more robust during exploration, especially under ε-greedy policies.
Expected Sarsa shows better average return across a wide range of step-sizes.
