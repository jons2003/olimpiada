# Improved Baseline for Adaptive Puzzle Solving

This solution improves the baseline with:
- Transformer-based value network (self-attention over cells)
- Larger training dataset (200k pairs, max_walk=120)
- A* with heuristic cache and batched expansion
- Parallel solving of instances (8 threads)

## Files
- `model.py` – ValueNet with Transformer encoder
- `search.py` – A* with cache
- `solve.py` – parallel A* for all instances
- `train.py` – collects backward walks and trains the model

## Usage
```bash
python train.py --time_limit 3000   # 50 minutes
python solve.py --time_limit 1500   # 25 minutes
python check.py
The same code works for any puzzle implementing the gym interface.