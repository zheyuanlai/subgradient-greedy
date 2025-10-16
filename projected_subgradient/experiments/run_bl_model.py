import torch
import sys
from pathlib import Path

# Ensure project root (directory containing 'projected_subgradient') is on sys.path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent  # simulation/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projected_subgradient.src.models.bernoulli_laplace import bernoulli_laplace_level_spectral
from projected_subgradient.src.algorithms.projected_subgradient import projected_subgradient_descent, compute_h
from projected_subgradient.src.utils.gpu_utils import get_device
import matplotlib.pyplot as plt
from projected_subgradient.src.utils.plotting import use_academic_style, finalize_axes, save_figure
import argparse
from projected_subgradient.logging.logger import ExperimentLogger as _ExpLogger
from pathlib import Path as _P
def _get_logger():
    return _ExpLogger(root_dir='projected_subgradient', model_name='bernoulli_laplace')
import os, json

def build_partition():
    # Fixed partition [[1,2],[3,5],[4]] -> zero-based [[0,1],[2,4],[3]]
    return [[0,1],[2,4],[3]]

def run_experiment(n=5, n_iter=None, s=1, epsilon=None, max_iter_cap=150, latex=False):
    """Run projected subgradient on Bernoulli-Laplace level model (restricted to {0,1}^5).
    Family {P_i}: powers of base chain to retain common stationary distribution.
    """
    d = 5

    device = get_device()
    print(f"Using device: {device}")
    print(f"State space size: 2^{d} = {2**d}")

    # Generate a set of transition matrices with different time scales
    P_list = []
    pi_list = []
    
    # For B-L model, create "different" matrices by taking powers
    # This simulates different time scales: P, P², P⁴, P⁸, P¹⁶
    print("\nBuilding base Bernoulli-Laplace level chain (shared stationary distribution)...")
    P_base, states, state_to_idx, pi_base = bernoulli_laplace_level_spectral(d, s=s, device=device)

    P_current = P_base
    for i in range(n):
        print(f"  P_{i+1}: P^{2**i}")
        P_list.append(P_current.clone())
        pi_list.append(pi_base)  # Stationary distribution is the same
        P_current = torch.matmul(P_current, P_current)  # Square the matrix

    # Partition selection
    partition = build_partition()
    
    print(f"\nFixed partition S = {partition}")

    # Initialize with uniform weights
    w_init = torch.ones(n, device=device) / n

    # Logger
    logger = _get_logger()
    logger.save_config({
        'model':'bernoulli_laplace_level_spectral', 'd':d, 'swap_size_s':s,
        'n':n, 'n_iter':n_iter, 'partition':partition
    })
    # Run with theoretical schedule (B computed internally) and recompute π̄
    # Determine iterations via theoretical bound if epsilon provided
    if epsilon is not None:
        # First do a cheap B estimation (uniform w via internal call with 1 iter dry run?)
        # We'll call algorithm with n_iter=1 and return_meta to get B without real progress.
        w_dummy = w_init.clone()
        _, _, _, meta = projected_subgradient_descent(
            w_dummy, P_list, pi_list, partition, d, states, state_to_idx,
            n_iter=1, use_theoretical_eta=True, recompute_pi_bar=True, verbose=False, return_meta=True
        )
        B_est = meta['B']
        t_needed = int((n * B_est) / (epsilon ** 2) + 0.999999)
        if n_iter is None:
            n_iter = t_needed
        print(f"Epsilon target ε={epsilon:.3e}: B≈{B_est:.3e} ⇒ theoretical iterations t=ceil(nB/ε²)={t_needed}")
        if n_iter > max_iter_cap:
            print(f"Capping iterations at max_iter_cap={max_iter_cap} (requested {n_iter})")
            n_iter = max_iter_cap
    if n_iter is None:
        n_iter = max_iter_cap
    print(f"\nRunning projected subgradient descent for {n_iter} iterations (theoretical η)...")
    print("="*70)
    w_optimal, w_avg, history = projected_subgradient_descent(
        w_init, P_list, pi_list, partition, d, states, state_to_idx,
        n_iter=n_iter, eta=None, B=None, use_theoretical_eta=True,
        recompute_pi_bar=True, verbose=True
    )
    for rec in history:
        logger.log_history_row(rec)

    print("\n" + "="*70)
    print("RESULTS:")
    print(f"Optimal weights w*: {w_optimal.cpu().numpy()}")
    print(f"Averaged weights w̄: {w_avg.cpu().numpy()}")
    print(f"Final h(w): {history[-1]['h(w)']:.6f}")
    
    # Compute final objective with optimal weights
    h_final = compute_h(w_optimal, P_list, pi_list, partition, d, states, state_to_idx)
    print(f"Objective h(w*) = {h_final.item():.6f}")

    # Plot convergence
    use_academic_style(latex=latex)
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 6.5))
    
    # Plot h(w) over iterations
    h_values = [h['h(w)'] for h in history]
    iterations = list(range(1, len(h_values)+1))
    axes[0].plot(iterations, h_values, marker='o', markersize=3.0, linewidth=1.6)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("h(w)")
    axes[0].set_title("Bernoulli-Laplace Level Model: Trajectory Plot of h(w)", pad=8)
    finalize_axes(axes[0])
    
    # Plot weight evolution
    for i in range(n):
        weights_i = [h['weights'][i] for h in history]
        axes[1].plot(iterations, weights_i, label=fr'$w_{{{i+1}}}$', linewidth=1.3, marker='o', markersize=2.4, markevery=max(1, n_iter//40))
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Weight")
    axes[1].set_title("Weights Evolution", pad=8)
    axes[1].legend(loc='center right', ncol=1, frameon=False, bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
    finalize_axes(axes[1])
    
    plot_path_png = logger.path('convergence.png')
    plot_path_pdf = logger.path('convergence.pdf')
    save_figure(fig, plot_path_png, plot_path_pdf)
    print(f"\nPlots saved to '{plot_path_png}' and PDF version.")
    logger.save_json('final', {
        'final_weights': w_optimal.cpu().tolist(),
        'average_weights': w_avg.cpu().tolist(),
        'final_h': history[-1]['h(w)']
    })
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of matrices (powers)")
    parser.add_argument("--iters", type=int, default=None, help="Override iterations (if not set and --epsilon provided, theoretical bound used)")
    parser.add_argument("--epsilon", type=float, default=None, help="Desired accuracy ε for iteration bound t=ceil(nB/ε²)")
    parser.add_argument("--s", type=int, default=1, help="Swap size parameter s")
    parser.add_argument("--max_iter_cap", type=int, default=200, help="Safety cap on iterations (reduced)")
    parser.add_argument("--latex", action='store_true', help="Enable LaTeX text rendering (requires TeX installation)")
    args = parser.parse_args()
    run_experiment(n=args.n, n_iter=args.iters, s=args.s, epsilon=args.epsilon, max_iter_cap=args.max_iter_cap, latex=args.latex)