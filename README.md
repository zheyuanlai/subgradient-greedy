# Information-theoretic minimax and submodular optimization algorithms for multivariate Markov chains

## Overview
This repository provides implementations and experiments for information‑theoretic optimization over multivariate Markov chains. We study projected subgradient methods and a two‑layer subgradient-greedy algorithm on Bernoulli–Laplace and Curie–Weiss models.

## Code Structure
```
/
├── projected_subgradient/   # Projected subgradient algorithm
├── psg_hd/                  # Optimized higher‑dimensional implementation
├── two_layer/               # Two‑layer subgradient-greedy algorithm
├── two_layer_hd/            # Optimized higher‑dimensional implementation
├── requirements.txt
└── README.md
```

## Numerical Experiments
We evaluate algorithms on multivariate Markov chains from the Bernoulli–Laplace level model and the Curie–Weiss model. Each module includes runner scripts and saves timestamped results (plots and logs) in its `results/` subfolder.

## Usage
1) Create an environment and install dependencies
```
conda create -n psg python=3.11 -y
conda activate psg
pip install -r requirements.txt
```
2) Run example experiments
```
# Projected subgradient algorithm
cd projected_subgradient
python experiments/run_bl_model.py
```

```
# Two‑layer subgradient-greedy algorithm
cd two_layer
python run_bl_two_layer.py
```

## Results
Artifacts are stored under `*/results/<model>/<timestamp>/` (plots, configs, logs). Results folders are ignored by Git by default.