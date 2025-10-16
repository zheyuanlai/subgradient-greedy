"""
Run high-d two-layer algorithm on the Bernoulli-Laplace level model (d up to ~10).
"""
import os
import sys
import json
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from pathlib import Path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projected_subgradient.src.utils.plotting import use_academic_style, finalize_axes
from two_layer_hd.algorithms.two_layer_hd import two_layer_algorithm_hd
from projected_subgradient.src.models.bernoulli_laplace import bernoulli_laplace_level_spectral
from projected_subgradient.src.utils.gpu_utils import get_device


def run_bl_two_layer_hd(
    d: int = 8,
    n_models: int = 5,
    l_cardinality: int = 7,
    K_inner: int = 150,
    eta: float | None = None,
    beta: float = 0.0,
    use_gpu: bool = True,
    save_results: bool = True,
    results_dir: str | None = None,
):
    device = get_device() if use_gpu else torch.device('cpu')
    print(f"Using device: {device}\n")

    print("Creating Bernoulli-Laplace level model...")
    P_base, states, state_to_idx, pi_base = bernoulli_laplace_level_spectral(d, s=1, device=device)

    # Family by squaring powers
    P_list, pi_list = [], []
    P_cur = P_base.clone()
    for _ in range(n_models):
        P_list.append(P_cur.clone())
        pi_list.append(pi_base)
        P_cur = P_cur @ P_cur
    print(f"Built {n_models} matrices of shape {P_base.shape}")

    V = [set([0, 1, 2, 3]), set([4, 5, 6])]

    if eta is None:
        B = max(100.0, float(d * d))
        eta = (n_models / (B * K_inner)) ** 0.5

    print("Configuration:")
    print(f"  d={d}, n={n_models}, l={l_cardinality}, K={K_inner}, eta={eta:.6f}")
    print(f"  V={V}, m={len(V)+1}\n")

    # Use consistent academic plotting style (mathtext)
    use_academic_style(latex=False)

    print("Running high-d two-layer algorithm...")
    print("=" * 60)
    S_final, w_final, f_final, history = two_layer_algorithm_hd(
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
    print("=" * 60, "\n", sep="")

    if save_results:
        if results_dir is None:
            results_dir = os.path.join(
                os.path.dirname(__file__),
                'results',
                'bernoulli_laplace_hd',
                datetime.now().strftime('%Y%m%d_%H%M%S')
            )
        os.makedirs(results_dir, exist_ok=True)

        config = {
            'model': 'bernoulli_laplace', 'd': d, 'n_models': n_models,
            'l_cardinality': l_cardinality, 'K_inner': K_inner,
            'eta': eta, 'beta': beta, 'V': [list(v) for v in V],
            'device': str(device)
        }
        with open(os.path.join(results_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        final_results = {
            'S_final': [list(s) for s in S_final],
            'w_final': w_final.cpu().tolist(),
            'f_final': f_final.item(),
            'supp_S_size': sum(len(s) for s in S_final)
        }
        with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        f_vals = history.get('f_values', [])
        axes[0].plot(range(1, len(f_vals)+1), f_vals, marker='o')
        axes[0].set_xlabel(r'Outer iteration $i$')
        axes[0].set_ylabel(r'$f(S_i, \bar{w}_i)$')
        axes[0].set_title(r'Outer objective $f$ over iterations')
        finalize_axes(axes[0])
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        inner = history.get('inner_f_values', [])
        for i, traj in enumerate(inner):
            axes[1].plot(range(1, len(traj)+1), traj, label=f'i={i+1}')
        axes[1].set_xlabel(r'Inner step $k$')
        axes[1].set_ylabel(r'$f(S_i, w^{(k)})$')
        axes[1].set_title(r'Inner PSG $f$ trajectories')
        finalize_axes(axes[1])
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(inner) > 0:
            axes[1].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'convergence.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(results_dir, 'convergence.pdf'), bbox_inches='tight')
        plt.close()
        print(f"Results saved to: {results_dir}\n")

    return S_final, w_final, f_final, history


if __name__ == "__main__":
    run_bl_two_layer_hd()
