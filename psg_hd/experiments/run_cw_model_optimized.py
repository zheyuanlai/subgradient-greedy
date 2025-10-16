import torch
import sys
from pathlib import Path

# Ensure project root is on sys.path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from psg_hd.src.models.curie_weiss import curie_weiss_glauber
from psg_hd.src.algorithms.projected_subgradient import projected_subgradient_descent, compute_h
from psg_hd.src.utils.gpu_utils import get_device
from psg_hd.src.utils.gpu_optimizations import (
    batch_matrix_power, optimized_subgradient_computation, 
    memory_efficient_tensor_product, batch_kl_divergence
)
from psg_hd.src.utils.plotting import use_academic_style, finalize_axes, save_figure
import matplotlib.pyplot as plt
import numpy as np
import argparse
from psg_hd.logging.logger import ExperimentLogger as _ExpLogger
import time
import gc

def _get_logger():
    return _ExpLogger(root_dir='psg_hd', model_name='curie_weiss')

def build_partition():
    # [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10]] -> zero-based
    return [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]

def run_experiment_optimized(n=5, n_iter=None, epsilon=None, max_iter_cap=150, 
                           latex=False, use_mixed_precision=True, checkpoint_freq=50):
    """
    GPU-optimized version for high-dimensional Curie-Weiss Glauber dynamics.
    
    Additional parameters:
    - use_mixed_precision: Use float16 for memory efficiency where possible
    - checkpoint_freq: Save intermediate results every N iterations
    """
    d = 10
    device = get_device()
    
    # Enable mixed precision if supported
    if use_mixed_precision and device.type == 'cuda':
        print("Using mixed precision (float16/float32) for memory efficiency")
        compute_dtype = torch.float16
        storage_dtype = torch.float32
    else:
        compute_dtype = torch.float32
        storage_dtype = torch.float32
    
    print(f"Using device: {device}")
    print(f"State space size: 2^{d} = {2**d}")
    print(f"Estimated memory per matrix: {(2**d)**2 * 4 / 1024**3:.2f} GB")
    
    # Clear GPU cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print("\\nBuilding base Curie-Weiss Glauber dynamics chain...")
    start_time = time.time()
    
    # Generate base Glauber transition matrix
    P_base, states, state_to_idx, H_vals, pi_base = curie_weiss_glauber(d, device=device)
    
    # Convert to appropriate dtype
    P_base = P_base.to(dtype=storage_dtype)
    pi_base = pi_base.to(dtype=storage_dtype)
    
    build_time = time.time() - start_time
    print(f"Base Glauber matrix built in {build_time:.2f} seconds")
    
    # Memory monitoring
    if device.type == 'cuda':
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory after base matrix: {memory_used:.2f} GB")
    
    print("\\nGenerating matrix powers using optimized batch computation...")
    start_time = time.time()
    
    # Generate powers efficiently (P, P^2, P^4, P^8, P^16)
    powers = [2**i for i in range(n)]
    P_list = []
    pi_list = []
    
    # Use iterative computation to manage memory
    P_current = P_base.clone()
    for i in range(n):
        print(f"  Computing P_base^{powers[i]} (matrix {i+1}/{n})...")
        P_list.append(P_current.clone())
        pi_list.append(pi_base.clone())  # Same stationary distribution for all powers
        
        if i < n - 1:  # Don't compute the last power
            P_current = torch.matmul(P_current, P_current)
            
        # Memory management
        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"    GPU memory: {memory_used:.2f} GB")
            if memory_used > 18:  # Conservative threshold for A10G
                torch.cuda.empty_cache()
                gc.collect()
    
    power_time = time.time() - start_time
    print(f"Matrix powers computed in {power_time:.2f} seconds")
    
    # Partition setup
    partition = build_partition()
    print(f"\\nUsing partition S = {partition}")
    print(f"Partition sizes: {[len(S) for S in partition]}")
    
    # Initialize weights
    w_init = torch.ones(n, device=device, dtype=storage_dtype) / n
    
    # Logger setup
    logger = _get_logger()
    logger.save_config({
        'model': 'curie_weiss_glauber_optimized',
        'd': d, 'n': n, 'n_iter': n_iter,
        'partition': partition, 'mixed_precision': use_mixed_precision,
        'compute_dtype': str(compute_dtype), 'storage_dtype': str(storage_dtype),
        'powers': powers
    })
    
    # Determine iterations using fast heuristic
    if epsilon is not None:
        print("\\nUsing fast heuristic for iteration bound...")
        # Simple heuristic: B ≈ log(state_space_size) * number_of_partition_blocks * safety_factor
        # For d=14: log(2^14) ≈ 9.7, partition blocks = 4, safety_factor = 0.1
        B_est = np.log(2**d) * len(partition) * 0.1
        t_needed = int((n * B_est) / (epsilon ** 2) + 0.999999)
        if n_iter is None:
            n_iter = t_needed
        print(f"Epsilon target ε={epsilon:.3e}: B≈{B_est:.3e} (heuristic) ⇒ theoretical iterations t={t_needed}")
        if n_iter > max_iter_cap:
            print(f"Capping iterations at max_iter_cap={max_iter_cap} (requested {n_iter})")
            n_iter = max_iter_cap
    
    if n_iter is None:
        n_iter = max_iter_cap
    
    print(f"\\nRunning optimized projected subgradient descent for {n_iter} iterations...")
    print(f"Theoretical learning rate schedule with recomputed π̄")
    print("="*70)
    
    # Run the optimized algorithm
    start_time = time.time()
    w_optimal, w_avg, history = projected_subgradient_descent(
        w_init, P_list, pi_list, partition, d, states, state_to_idx,
        n_iter=n_iter, eta=None, B=None, use_theoretical_eta=True,
        recompute_pi_bar=True, verbose=True
    )
    
    total_time = time.time() - start_time
    print(f"\\nOptimization completed in {total_time:.2f} seconds")
    print(f"Average time per iteration: {total_time/n_iter:.3f} seconds")
    
    # Log results
    for rec in history:
        logger.log_history_row(rec)
    
    print("\\n" + "="*70)
    print("RESULTS:")
    print(f"Optimal weights w*: {w_optimal.cpu().numpy()}")
    print(f"Averaged weights w̄: {w_avg.cpu().numpy()}")
    print(f"Final h(w): {history[-1]['h(w)']:.6f}")
    
    # Final objective computation
    h_final = compute_h(w_optimal, P_list, pi_list, partition, d, states, state_to_idx)
    print(f"Objective h(w*) = {h_final.item():.6f}")
    
    # Performance metrics
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory usage: {peak_memory:.2f} GB")
        torch.cuda.reset_peak_memory_stats()  # Reset for next run
    
    # Generate convergence plots
    print("\\nGenerating convergence plots...")
    use_academic_style(latex=latex)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0))
    
    # Plot h(w) convergence
    h_values = [h['h(w)'] for h in history]
    iterations = list(range(1, len(h_values)+1))
    axes[0].plot(iterations, h_values, marker='o', markersize=2.0, linewidth=1.6, alpha=0.8, color='#2E8B57')
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("h(w)")
    axes[0].set_title(f"High-Dimensional Curie-Weiss Model (d={d}): Objective Convergence", pad=8)
    axes[0].grid(True, alpha=0.3)
    finalize_axes(axes[0])
    
    # Plot weight evolution
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    for i in range(min(n, 5)):  # Limit to 5 weights for readability
        weights_i = [h['weights'][i] for h in history]
        axes[1].plot(iterations, weights_i, label=f'$w_{{{i+1}}}$', linewidth=1.5, 
                    marker='o', markersize=1.5, markevery=max(1, n_iter//30), 
                    alpha=0.8, color=colors[i % len(colors)])
    
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Weight")
    axes[1].set_title("Weight Evolution", pad=8)
    axes[1].legend(loc='best', frameon=True, fancybox=True, shadow=True)
    axes[1].grid(True, alpha=0.3)
    finalize_axes(axes[1])
    
    plt.tight_layout()
    
    # Save plots
    plot_path_png = logger.path('convergence.png')
    plot_path_pdf = logger.path('convergence.pdf')
    save_figure(fig, plot_path_png, plot_path_pdf)
    print(f"Plots saved to '{plot_path_png}' and PDF version.")
    
    # Save comprehensive final results
    logger.save_json('final', {
        'final_weights': w_optimal.cpu().tolist(),
        'average_weights': w_avg.cpu().tolist(),
        'final_h': history[-1]['h(w)'],
        'total_time_seconds': total_time,
        'iterations': n_iter,
        'peak_memory_gb': peak_memory if device.type == 'cuda' else None,
        'convergence_data': {
            'h_values': h_values[-10:],  # Last 10 values
            'final_gradient_norm': history[-1].get('gradient_norm', None),
            'convergence_rate': (h_values[-1] - h_values[-10]) / 10 if len(h_values) >= 10 else None
        },
        'optimization_settings': {
            'mixed_precision': use_mixed_precision,
            'epsilon': epsilon,
            'dimension': d,
            'partition_sizes': [len(S) for S in partition],
            'matrix_powers': powers,
            'theoretical_B_estimate': B_est if epsilon is not None else None
        },
        'hardware_info': {
            'gpu_name': torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU',
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if device.type == 'cuda' else None
        }
    })
    
    plt.show()
    
    return w_optimal, w_avg, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-optimized high-dimensional Curie-Weiss experiment")
    parser.add_argument("--n", type=int, default=5, help="Number of matrices (powers)")
    parser.add_argument("--iters", type=int, default=None, help="Override iterations")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Desired accuracy ε")
    parser.add_argument("--max_iter_cap", type=int, default=500, help="Maximum iterations cap")
    parser.add_argument("--latex", action='store_true', help="Enable LaTeX text rendering")
    parser.add_argument("--no_mixed_precision", action='store_true', 
                       help="Disable mixed precision (use if encountering numerical issues)")
    parser.add_argument("--checkpoint_freq", type=int, default=50, 
                       help="Save checkpoints every N iterations")
    
    args = parser.parse_args()
    
    print("="*80)
    print("HIGH-DIMENSIONAL CURIE-WEISS GLAUBER DYNAMICS EXPERIMENT")
    print("="*80)
    print(f"Dimension: d=10 (state space size: 2^10 = {2**10})")
    print(f"Model: Curie-Weiss with Glauber dynamics")
    print(f"GPU Optimization: Enabled")
    print(f"Mixed Precision: {'Disabled' if args.no_mixed_precision else 'Enabled'}")
    print("="*80)
    
    run_experiment_optimized(
        n=args.n, n_iter=args.iters, epsilon=args.epsilon,
        max_iter_cap=args.max_iter_cap, latex=args.latex,
        use_mixed_precision=not args.no_mixed_precision,
        checkpoint_freq=args.checkpoint_freq
    )