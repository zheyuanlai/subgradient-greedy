"""
Analyze h(w) values from projected subgradient experiments.

This script computes various h(w) values for experiments on both Bernoulli-Laplace
and Curie-Weiss models to create a comparison table:
- h(w_opt): the running minimum from the algorithm
- h(w̄): computed from the average weights over all iterations
- h(w_uniform): computed with uniform weights
- h(w_extreme): computed with extreme weights (1, 0, 0, ..., 0)
- h(w_final): the final h value from the last iteration
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from psg_hd.src.models.bernoulli_laplace import bernoulli_laplace_level_spectral
from psg_hd.src.models.curie_weiss import curie_weiss_glauber
from psg_hd.src.algorithms.projected_subgradient_optimized import compute_h
from psg_hd.src.utils.gpu_utils import get_device


def load_experiment_data(result_dir):
    """Load configuration and history from an experiment directory."""
    config_path = result_dir / "config.json"
    history_path = result_dir / "history.csv"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    history = pd.read_csv(history_path)
    
    return config, history


def parse_weights_from_csv(weights_str):
    """Parse weight vector from CSV string format."""
    return np.array([float(x) for x in weights_str.strip().split()])


def build_partition(d):
    """Build FULL partition that covers all d coordinates."""
    if d == 5:
        return [[0, 1], [2, 4], [3]]  # Full partition for d=5
    elif d == 8:
        return [[0, 1, 2], [3, 4], [5, 6, 7]]  # Full partition for d=8
    elif d == 10:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]  # Full partition for d=10
    elif d == 12:
        return [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9, 10, 11]]  # Full partition for d=12
    elif d == 15:
        return [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14]]  # Full partition for d=15
    else:
        # Generic: split into chunks of ~3, ensuring all coordinates covered
        chunk_size = max(3, d // 5)
        return [list(range(i, min(i + chunk_size, d))) for i in range(0, d, chunk_size)]


def setup_bernoulli_laplace_model(d, s, n, device):
    """Build Bernoulli-Laplace model with n powers of the base matrix."""
    P_base, states, state_to_idx, pi_base = bernoulli_laplace_level_spectral(d, s=s, device=device)
    
    P_list = []
    pi_list = []
    P_current = P_base
    
    for i in range(n):
        P_list.append(P_current.clone())
        pi_list.append(pi_base)
        P_current = torch.matmul(P_current, P_current)  # Square the matrix
    
    return P_list, pi_list, states, state_to_idx


def setup_curie_weiss_model(d, n, device):
    """Build Curie-Weiss model with n powers of the base matrix."""
    P_base, states, state_to_idx, H_vals, pi_base = curie_weiss_glauber(d, device=device)
    
    P_list = []
    pi_list = []
    P_current = P_base.clone()
    
    for i in range(n):
        P_list.append(P_current.clone())
        pi_list.append(pi_base)
        P_current = torch.matmul(P_current, P_current)  # Square the matrix
    
    return P_list, pi_list, states, state_to_idx


def analyze_experiment(result_dir, model_type, device):
    """Analyze a single experiment and compute all h(w) values."""
    print(f"\nAnalyzing: {result_dir.name}")
    print("=" * 70)
    
    # Load experiment data
    config, history = load_experiment_data(result_dir)
    
    d = config.get('d', 5)
    n = config.get('n', 5)
    
    # Skip extremely high-dimensional experiments for practicality
    if d > 15:
        print(f"Skipping very high-dimensional experiment (d={d}) - too slow for analysis")
        return None
    
    partition = build_partition(d)
    
    # Setup model
    if model_type == 'bernoulli_laplace':
        s = config.get('swap_size_s', 1)
        P_list, pi_list, states, state_to_idx = setup_bernoulli_laplace_model(d, s, n, device)
    elif model_type == 'curie_weiss':
        P_list, pi_list, states, state_to_idx = setup_curie_weiss_model(d, n, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Parse weights from history
    all_weights = []
    for weights_str in history['weights']:
        w = parse_weights_from_csv(weights_str)
        all_weights.append(w)
    all_weights = np.array(all_weights)
    
    # Compute various h(w) values
    
    # 1. h(w_opt): running minimum
    h_values = history['h'].values
    h_opt = np.min(h_values)
    opt_idx = np.argmin(h_values)
    w_opt = torch.tensor(all_weights[opt_idx], dtype=torch.float32, device=device)
    
    # 2. h(w̄): average weights over all iterations
    w_avg = np.mean(all_weights, axis=0)
    w_avg_tensor = torch.tensor(w_avg, dtype=torch.float32, device=device)
    print(f"Computing h(w̄) for d={d} (state space: 2^{d} = {2**d})...")
    start_time = time.time()
    h_avg = compute_h(w_avg_tensor, P_list, pi_list, partition, d, states, state_to_idx).item()
    avg_time = time.time() - start_time
    print(f"h(w̄) computed in {avg_time:.2f} seconds")
    
    # Memory management
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # 3. h(w_uniform): uniform weights
    w_uniform = torch.ones(n, device=device) / n
    print("Computing h(w_uniform)...")
    start_time = time.time()
    h_uniform = compute_h(w_uniform, P_list, pi_list, partition, d, states, state_to_idx).item()
    uniform_time = time.time() - start_time
    print(f"h(w_uniform) computed in {uniform_time:.2f} seconds")
    
    # Memory management
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # 4. h(w_extreme): extreme weights (1, 0, 0, ..., 0)
    w_extreme = torch.zeros(n, device=device)
    w_extreme[0] = 1.0
    print("Computing h(w_extreme)...")
    start_time = time.time()
    h_extreme = compute_h(w_extreme, P_list, pi_list, partition, d, states, state_to_idx).item()
    extreme_time = time.time() - start_time
    print(f"h(w_extreme) computed in {extreme_time:.2f} seconds")
    
    # Memory management
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # 5. h(w_final): final iteration value
    h_final = h_values[-1]
    w_final = torch.tensor(all_weights[-1], dtype=torch.float32, device=device)
    
    # Print results
    print(f"Model: {config.get('model', 'unknown')}")
    print(f"Configuration: d={d}, n={n}, partition={partition}")
    print(f"Total iterations: {len(history)}")
    print()
    print(f"h(w_opt)     = {h_opt:.8f}  (iteration {opt_idx + 1})")
    print(f"h(w̄)        = {h_avg:.8f}  (average weights)")
    print(f"h(w_uniform) = {h_uniform:.8f}  (uniform weights)")
    print(f"h(w_extreme) = {h_extreme:.8f}  (extreme: [1, 0, ..., 0])")
    print(f"h(w_final)   = {h_final:.8f}  (final iteration)")
    print()
    print(f"w_opt     = {w_opt.cpu().numpy()}")
    print(f"w̄        = {w_avg}")
    print(f"w_uniform = {w_uniform.cpu().numpy()}")
    print(f"w_extreme = {w_extreme.cpu().numpy()}")
    print(f"w_final   = {w_final.cpu().numpy()}")
    
    return {
        'experiment': result_dir.name,
        'model': config.get('model', 'unknown'),
        'd': d,
        'n': n,
        'iterations': len(history),
        'h_opt': h_opt,
        'h_avg': h_avg,
        'h_uniform': h_uniform,
        'h_extreme': h_extreme,
        'h_final': h_final,
        'w_opt': w_opt.cpu().numpy().tolist(),
        'w_avg': w_avg.tolist(),
        'w_uniform': w_uniform.cpu().numpy().tolist(),
        'w_extreme': w_extreme.cpu().numpy().tolist(),
        'w_final': w_final.cpu().numpy().tolist(),
    }


def analyze_all_experiments(model_name):
    """Analyze all experiments for a given model."""
    device = get_device()
    print(f"Using device: {device}")
    
    results_dir = Path(__file__).parent / "psg_hd" / "results" / model_name
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return []
    
    # Get all experiment directories
    exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    
    if not exp_dirs:
        print(f"No experiment directories found in {results_dir}")
        return []
    
    print(f"\nFound {len(exp_dirs)} experiments for {model_name}")
    
    results = []
    for exp_dir in exp_dirs:
        try:
            result = analyze_experiment(exp_dir, model_name, device)
            if result is not None:  # Skip high-dimensional experiments
                results.append(result)
        except Exception as e:
            print(f"Error analyzing {exp_dir.name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def create_summary_table(results):
    """Create a summary table from analysis results."""
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    # Convert weight lists to string format for CSV
    df['w_opt_str'] = df['w_opt'].apply(lambda x: ' '.join([f'{v:.6f}' for v in x]))
    df['w_avg_str'] = df['w_avg'].apply(lambda x: ' '.join([f'{v:.6f}' for v in x]))
    df['w_uniform_str'] = df['w_uniform'].apply(lambda x: ' '.join([f'{v:.6f}' for v in x]))
    df['w_extreme_str'] = df['w_extreme'].apply(lambda x: ' '.join([f'{v:.6f}' for v in x]))
    df['w_final_str'] = df['w_final'].apply(lambda x: ' '.join([f'{v:.6f}' for v in x]))
    
    # Select columns for display (including all weight vectors)
    display_cols = ['experiment', 'model', 'd', 'n', 'iterations', 
                    'h_opt', 'w_opt_str',
                    'h_avg', 'w_avg_str', 
                    'h_uniform', 'w_uniform_str',
                    'h_extreme', 'w_extreme_str',
                    'h_final', 'w_final_str']
    
    # Rename for cleaner headers
    df_display = df[display_cols].copy()
    df_display.rename(columns={
        'w_opt_str': 'w_opt',
        'w_avg_str': 'w_avg',
        'w_uniform_str': 'w_uniform',
        'w_extreme_str': 'w_extreme',
        'w_final_str': 'w_final'
    }, inplace=True)
    
    return df_display


def main():
    """Main analysis function."""
    print("=" * 70)
    print("PSG_HD PROJECTED SUBGRADIENT EXPERIMENT ANALYSIS")
    print("=" * 70)
    
    # Analyze Bernoulli-Laplace experiments
    print("\n\nBERNOULLI-LAPLACE MODEL")
    print("=" * 70)
    bl_results = analyze_all_experiments('bernoulli_laplace')
    
    if bl_results:
        print("\n\nSUMMARY TABLE: BERNOULLI-LAPLACE")
        print("=" * 70)
        bl_table = create_summary_table(bl_results)
        print(bl_table.to_string(index=False))
        
        # Save to CSV
        output_path = Path(__file__).parent / "psg_hd" / "results" / "bernoulli_laplace_analysis.csv"
        bl_table.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
    
    # Analyze Curie-Weiss experiments
    print("\n\n\nCURIE-WEISS MODEL")
    print("=" * 70)
    cw_results = analyze_all_experiments('curie_weiss')
    
    if cw_results:
        print("\n\nSUMMARY TABLE: CURIE-WEISS")
        print("=" * 70)
        cw_table = create_summary_table(cw_results)
        print(cw_table.to_string(index=False))
        
        # Save to CSV
        output_path = Path(__file__).parent / "psg_hd" / "results" / "curie_weiss_analysis.csv"
        cw_table.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
    
    # Combined comparison if both available
    if bl_results and cw_results:
        print("\n\n\nCOMBINED COMPARISON")
        print("=" * 70)
        all_results = bl_results + cw_results
        combined_table = create_summary_table(all_results)
        print(combined_table.to_string(index=False))
        
        # Save combined
        output_path = Path(__file__).parent / "psg_hd" / "results" / "combined_analysis.csv"
        combined_table.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()