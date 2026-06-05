# Current Training Results (Member B)

This document summarizes the real-time results of the ongoing `5000`-episode multi-seed experiment for the **Member B DQN Pipeline**.

## General Setup
- **Algorithm**: Deep Q-Network (DQN) with Action Masking
- **State Space**: Flattened Board (40) + Remaining Piece Counts (10)
- **Reward Function**: Sparse (+1 per placement, +100 for completion)
- **Epsilon Schedule**: Decays from $1.0 \rightarrow 0.05$ between episodes 200 and 2200.

## Seed Progress

**1. Seed 42: Completed ✅**
- Total Episodes: 5,000
- Final Epsilon: 0.136
- Final Mean Reward (last 100 eps): ~7.71
- Final Mean Coverage (last 100 eps): ~77.1%
- **Successes Observed**: The agent successfully solved the board at least twice (observed at episode 3,500 and episode 4,800), briefly spiking the trailing 100-episode success rate to `1.0%`.

**2. Seed 123: Running / Nearing Completion 🔄**
- Current Episode: 4,900
- Mean Reward (last 100 eps): ~7.60
- Mean Coverage (last 100 eps): ~76.0%
- **Successes Observed**: 0 successes observed within the trailing 100-episode windows so far.

**3. Seeds 456, 789, 1024: Pending ⏳**
- Will automatically execute sequentially in the background runner.

## Preliminary Analysis
The sparse reward function makes discovering the optimal tangram configuration extremely difficult due to the massive search space. However, we have successfully verified that the DQN agent *can* solve the board entirely, as evidenced by the hits in Seed 42.

Because epsilon decay limits exploration heavily in the later stages (decaying to ~0.15 by episode 4500), the agent converges on strong suboptimal strategies (achieving ~77% board coverage consistently). To achieve a high, robust success rate, the model requires the full `500,000+` episodes requested in the assignment outline, allowing for extended exploration and a much wider replay buffer.

> *Note: The full learning curves will be generated using `python -m member_goktug.plot_curves` once all seeds finish.*
