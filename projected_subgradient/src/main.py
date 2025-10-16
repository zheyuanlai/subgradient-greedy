import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
# This allows imports from src, experiments, etc.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import run_bl_model, run_cw_model

def main():
    parser = argparse.ArgumentParser(description="Run numerical experiments for projected subgradient algorithm.")
    parser.add_argument('model', choices=['bl', 'cw'], help="The model to run: 'bl' for Bernoulli-Laplace, 'cw' for Curie-Weiss.")
    
    args = parser.parse_args()

    if args.model == 'bl':
        print("Running Bernoulli-Laplace model experiment...")
        run_bl_model.run_experiment()
    elif args.model == 'cw':
        print("Running Curie-Weiss model experiment...")
        run_cw_model.run_experiment()

if __name__ == "__main__":
    main()