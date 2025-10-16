"""
Run two-layer algorithm on Bernoulli-Laplace model.
"""
import torch
import sys
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from pathlib import Path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projected_subgradient.src.utils.plotting import use_academic_style, finalize_axes
from two_layer.algorithms.two_layer import two_layer_algorithm
from projected_subgradient.src.models.bernoulli_laplace import bernoulli_laplace_level_spectral
from projected_subgradient.src.utils.gpu_utils import get_device


def run_bl_two_layer_experiment(
    d: int = 5,
    n_models: int = 3,
    l_cardinality: int = 10,
    K_inner: int = 100,
    eta: float = None,
    beta: float = 0.0,
    use_gpu: bool = True,
    save_results: bool = True,
    results_dir: str = None
):
    """
    Run two-layer algorithm experiment on Bernoulli-Laplace model.
    
    Args:
        d: Dimension of state space
        n_models: Number of transition matrices
        l_cardinality: Cardinality constraint
        K_inner: Number of inner loop iterations
        eta: Step size (if None, computed as sqrt(n/(B*K)))
        beta: Constant for g and c functions
        use_gpu: Whether to use GPU
        save_results: Whether to save results
        results_dir: Directory to save results
    """
    # Setup device
    device = get_device() if use_gpu else torch.device('cpu')
    print(f"Using device: {device}")
    print()
    
    # Create Bernoulli-Laplace base model and derive a family by squaring powers
    print("Creating Bernoulli-Laplace level model (spectral)...")
    P_base, states, state_to_idx, pi_base = bernoulli_laplace_level_spectral(d, s=1, device=device)
    # Build family P_i = P_base^{2^{i-1}}
    P_list = []
    pi_list = []
    P_current = P_base.clone()
    for i in range(n_models):
        P_list.append(P_current.clone())
        pi_list.append(pi_base)
        P_current = P_current @ P_current
    print(f"Created {n_models} transition matrices of size {P_base.shape[0]}x{P_base.shape[1]}")
    print()
    
    V = [{0, 1}, {2, 4}]
    
    # Compute step size if not provided
    if eta is None:
        # From the manuscript: eta = sqrt(n / (B * K))
        # B is the upper bound on the objective, estimate as d * log(2^d) = d^2 * log(2)
        B = d * d * torch.log(torch.tensor(2.0))
        eta = torch.sqrt(torch.tensor(n_models / (B * K_inner))).item()
    
    print(f"Configuration:")
    print(f"  d (dimension): {d}")
    print(f"  n (models): {n_models}")
    print(f"  l (cardinality): {l_cardinality}")
    print(f"  K (inner iterations): {K_inner}")
    print(f"  eta (step size): {eta:.6f}")
    print(f"  V (partition): {V}")
    print(f"  m (partitions): {len(V) + 1}")
    print()
    
    # Use readable plotting style
    use_academic_style(latex=False)

    # Run two-layer algorithm
    print("Running two-layer algorithm...")
    print("=" * 60)
    S_final, w_final, f_final, history = two_layer_algorithm(
        P_list=P_list,
        pi_list=pi_list,
        d=d,
        states=states,
        state_to_idx=state_to_idx,
        V=V,
        l=l_cardinality,
        K=K_inner,
        eta=eta,
        beta=beta,
        verbose=True,
        return_history=True,
    )
    print("=" * 60)
    print()
    
    # Save results
    if save_results:
        if results_dir is None:
            results_dir = os.path.join(
                os.path.dirname(__file__), 
                'results', 
                'bernoulli_laplace',
                datetime.now().strftime('%Y%m%d_%H%M%S')
            )
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Save configuration
        config = {
            'model': 'bernoulli_laplace',
            'd': d,
            'n_models': n_models,
            'l_cardinality': l_cardinality,
            'K_inner': K_inner,
            'eta': eta,
            'beta': beta,
            'V': [list(v) for v in V],
            'device': str(device)
        }
        
        with open(os.path.join(results_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save final results
        final_results = {
            'S_final': [list(s) for s in S_final],
            'w_final': w_final.cpu().tolist(),
            'f_final': f_final.item(),
            'supp_S_size': sum(len(s) for s in S_final)
        }
        
        with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Save history
        # Prepare JSON-serializable history fields
        s_before = [[list(s) for s in S_iter] for S_iter in history.get('S_before', [])]
        s_after = [[list(s) for s in S_iter] for S_iter in history.get('S_after', [])]
        w_before = [w.cpu().tolist() for w in history.get('w_before_psg', [])]
        w_after = [w.cpu().tolist() for w in history.get('w_after_psg', [])]
        history_data = {
            'f_values': history.get('f_values', []),
            'inner_f_values': history.get('inner_f_values', []),
            'S_before': s_before,
            'S_after': s_after,
            'w_before_psg': w_before,
            'w_after_psg': w_after,
        }
        
        with open(os.path.join(results_dir, 'history.json'), 'w') as f:
            json.dump(history_data, f, indent=2)
        
        # Plot convergence (only f): outer iterations and inner PSG trajectories
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # Outer f per iteration
        f_vals = history.get('f_values', [])
        axes[0].plot(range(1, len(f_vals) + 1), f_vals, marker='o')
        axes[0].set_xlabel(r'Outer iteration $i$')
        axes[0].set_ylabel(r'$f(S_i, \bar{w}_i)$')
        axes[0].set_title(r'Outer objective $f$ over iterations')
        finalize_axes(axes[0])
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        # Inner f trajectories
        inner = history.get('inner_f_values', [])
        for i, traj in enumerate(inner):
            axes[1].plot(range(1, len(traj)+1), traj, label=f'i={i+1}')
        axes[1].set_xlabel(r'Inner step $k$')
        axes[1].set_ylabel(r'$f(S_i, w^{(k)})$')
        axes[1].set_title(r'Inner PSG $f$ trajectories')
        finalize_axes(axes[1])
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(inner) > 0:
            axes[1].legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'convergence.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(results_dir, 'convergence.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"Results saved to: {results_dir}")
        print()
    
    return S_final, w_final, f_final, history


if __name__ == "__main__":
    # Run experiment
    S, w, f, history = run_bl_two_layer_experiment(
        d=5,
        n_models=5,
        l_cardinality=4,
        K_inner=150,
        use_gpu=True,
        save_results=True
    )
    
    print("\nExperiment completed successfully!")
