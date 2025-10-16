"""
Run projected subgradient on Curie-Weiss model with OPTIMIZED tensor product.

This version uses compute_tensor_product_vectorized for d=10 experiments.
Matches the structure and output format of projected_subgradient/experiments/run_cw_model.py
"""
import torch
import sys
from pathlib import Path

# Ensure project root is on sys.path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent  # simulation/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from psg_hd.src.models.curie_weiss import curie_weiss_glauber
from psg_hd.src.algorithms.projected_subgradient_optimized import projected_subgradient_descent, compute_h
from psg_hd.src.utils.gpu_utils import get_device
import matplotlib.pyplot as plt
from psg_hd.src.utils.plotting import use_academic_style, finalize_axes, save_figure
import argparse
from psg_hd.custom_logging.logger import ExperimentLogger as _ExpLogger
from pathlib import Path as _P
def _get_logger():
    return _ExpLogger(root_dir='psg_hd', model_name='curie_weiss')
import os, json


def build_partition(d):
    """Build FULL partition that covers all d coordinates."""
    if d == 5:
        return [[0, 1], [2, 4], [3]]  # Full partition for d=5
    elif d == 8:
        return [[0, 1, 2], [3, 4], [5, 6, 7]]  # Full partition for d=8
    elif d == 10:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]  # Full partition for d=10
    elif d == 15:
        return [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14]]  # Full partition for d=15
    else:
        # Generic: split into chunks of ~3, ensuring all coordinates covered
        chunk_size = max(3, d // 5)
        return [list(range(i, min(i + chunk_size, d))) for i in range(0, d, chunk_size)]


def run_experiment(d=10, n=5, n_iter=None, T=10.0, h_param=1.0, epsilon=None, max_iter_cap=300, latex=False, force_cpu=False):
    """
    Run optimized projected subgradient on Curie-Weiss Glauber dynamics.
    
    Args:
        d: Dimension (default 10 for high-dimensional experiments)
        n: Number of models in family
        n_iter: Number of iterations (overrides epsilon if provided)
        T: Temperature parameter
        h_param: External field parameter
        epsilon: Desired accuracy ε for iteration bound
        max_iter_cap: Maximum iterations cap
        latex: Enable LaTeX rendering in plots
    """
    # Select device (allow forcing CPU via flag)
    device = torch.device('cpu') if force_cpu else get_device()
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"State space size: 2^{d} = {2**d}")

    # Build family of matrices: P, P², P⁴, P⁸, P¹⁶, ...
    print(f"\nBuilding base Glauber transition matrix (shared stationary distribution)...")
    P_base, states, state_to_idx, H_vals, pi_base = curie_weiss_glauber(d, T=T, h=h_param, device=device)
    P_list = []
    pi_list = []
    P_current = P_base.clone()
    # First n-1 are powers of P_base
    for i in range(n - 1):
        P_list.append(P_current.clone())
        pi_list.append(pi_base)
        P_current = P_current @ P_current
    
    # The last one is the lazy chain
    I = torch.eye(P_base.shape[0], device=device)
    P_lazy = 0.5 * (I + P_base)
    P_list.append(P_lazy)
    pi_list.append(pi_base) # The lazy chain has the same stationary distribution

    print(f"Created {n} transition matrices of size {P_base.shape[0]}x{P_base.shape[1]}")
    print()

    # CRITICAL: Use FULL partition that covers all coordinates
    partition = build_partition(d)
    print(f"\nFixed partition S = {partition}")

    # Initialize with uniform weights
    w_init = torch.ones(n, device=device) / n

    # Logger
    logger = _get_logger()
    logger.save_config({
        'model': 'curie_weiss_glauber',
        'd': d,
        'T': T,
        'h': h_param,
        'n': n,
        'n_iter': n_iter,
        'partition': partition,
        'optimization': 'vectorized_tensor_product'
    })

    # Determine iterations via theoretical bound if epsilon provided
    if epsilon is not None:
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
    
    print(f"\nRunning OPTIMIZED projected subgradient descent for {n_iter} iterations (theoretical η)...")
    print("=" * 70)
    
    w_optimal, w_avg, history = projected_subgradient_descent(
        w_init, P_list, pi_list, partition, d, states, state_to_idx,
        n_iter=n_iter, eta=None, B=None, use_theoretical_eta=True,
        recompute_pi_bar=True, verbose=True
    )
    
    for rec in history:
        logger.log_history_row(rec)

    print("\n" + "=" * 70)
    print("RESULTS:")
    print(f"Optimal weights w*: {w_optimal.cpu().numpy()}")
    print(f"Averaged weights w̄: {w_avg.cpu().numpy()}")
    print(f"Final h(w): {history[-1]['h(w)']:.6f}")
    
    # Compute final objective with optimal weights
    h_final = compute_h(w_optimal, P_list, pi_list, partition, d, states, state_to_idx)
    print(f"Objective h(w*) = {h_final.item():.6f}")

    # Plot convergence (matching d=5 style)
    use_academic_style(latex=latex)
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 6.5))
    
    # Plot h(w) over iterations
    h_values = [h['h(w)'] for h in history]
    iterations = list(range(1, len(h_values) + 1))
    axes[0].plot(iterations, h_values, marker='o', markersize=3.0, linewidth=1.6)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("h(w)")
    axes[0].set_title(f"Curie-Weiss Model (d={d}): Trajectory Plot of h(w)", pad=8)
    finalize_axes(axes[0])
    
    # Plot weight evolution
    for i in range(n):
        weights_i = [h['weights'][i] for h in history]
        axes[1].plot(iterations, weights_i, label=fr'$w_{{{i+1}}}$', 
                    linewidth=1.3, marker='o', markersize=2.4, markevery=max(1, n_iter//40))
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Weight")
    axes[1].set_title("Weights Evolution", pad=8)
    axes[1].legend(loc='center right', ncol=1, frameon=False, 
                   bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
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
    parser = argparse.ArgumentParser(description='Run optimized PSG on Curie-Weiss model')
    parser.add_argument("--d", type=int, default=10, help="Dimension (default: 10)")
    # Accept both --n and --n_models for convenience
    parser.add_argument("--n", "--n_models", dest="n", type=int, default=5,
                        help="Number of matrices (powers) (default: 5)")
    parser.add_argument("--iters", type=int, default=None, 
                       help="Override iterations (if not set and --epsilon provided, theoretical bound used)")
    parser.add_argument("--T", type=float, default=10.0, help="Temperature parameter (default: 10.0)")
    parser.add_argument("--h", type=float, default=1.0, help="External field parameter (default: 1.0)")
    parser.add_argument("--epsilon", type=float, default=None, 
                       help="Desired accuracy ε for iteration bound t=ceil(nB/ε²)")
    # Accept both --max_iter and --max_iter_cap
    parser.add_argument("--max_iter", "--max_iter_cap", dest="max_iter_cap", type=int, default=300, 
                       help="Safety cap on iterations (default: 300)")
    parser.add_argument("--latex", action='store_true', 
                       help="Enable LaTeX text rendering (requires TeX installation)")
    parser.add_argument("--cpu", action='store_true', help="Force CPU execution even if CUDA is available")
    
    args = parser.parse_args()
    run_experiment(d=args.d, n=args.n, n_iter=args.iters, T=args.T, h_param=args.h,
                  epsilon=args.epsilon, max_iter_cap=args.max_iter_cap, latex=args.latex, force_cpu=args.cpu)
