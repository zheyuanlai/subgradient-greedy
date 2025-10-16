"""
Performance benchmark comparing original vs optimized tensor product computation.

Run this to verify the optimization works and measure speedup.
"""
import torch
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.bernoulli_laplace import bernoulli_laplace_level_spectral
from src.algorithms.projected_subgradient import compute_tensor_product as compute_tensor_product_original
from src.algorithms.projected_subgradient_optimized import compute_tensor_product_vectorized


def benchmark_tensor_product(d, num_runs=3, device='cpu'):
    """
    Benchmark original vs optimized tensor product computation.
    
    Args:
        d: Dimension
        num_runs: Number of runs to average
        device: 'cpu' or 'cuda'
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking d={d} (state space size = {2**d})")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Create model
    device_obj = torch.device(device)
    P, states, state_to_idx, pi = bernoulli_laplace_level_spectral(d, s=1, device=device_obj)
    
    # Define partition - MUST be FULL (cover all coordinates)!
    if d == 5:
        partition = [[0, 1], [2, 4], [3]]  # Full partition for d=5
    elif d == 10:
        partition = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]  # Full partition for d=10
    else:
        # Generic partition: split into chunks of 3 (automatically full)
        partition = [list(range(i, min(i+3, d))) for i in range(0, d, 3)]
    
    print(f"Partition: {partition}")
    print(f"Number of partition elements: {len(partition)}")
    print()
    
    # Warm up GPU if using CUDA
    if device == 'cuda':
        _ = compute_tensor_product_vectorized(P, pi, partition, d, states, state_to_idx)
        torch.cuda.synchronize()
    
    # Benchmark original version (skip if d > 8 to avoid long wait)
    if d <= 8:
        print("Testing ORIGINAL implementation...")
        times_original = []
        for run in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            Q_original = compute_tensor_product_original(P, pi, partition, d, states, state_to_idx)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            elapsed = end - start
            times_original.append(elapsed)
            print(f"  Run {run+1}: {elapsed:.4f} seconds")
        
        avg_original = sum(times_original) / len(times_original)
        print(f"  Average: {avg_original:.4f} seconds")
        print()
    else:
        print("Skipping ORIGINAL implementation (d > 8, would take too long)")
        Q_original = None
        avg_original = None
        print()
    
    # Benchmark optimized version
    print("Testing OPTIMIZED implementation...")
    times_optimized = []
    for run in range(num_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        Q_optimized = compute_tensor_product_vectorized(P, pi, partition, d, states, state_to_idx)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        elapsed = end - start
        times_optimized.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.4f} seconds")
    
    avg_optimized = sum(times_optimized) / len(times_optimized)
    print(f"  Average: {avg_optimized:.4f} seconds")
    print()
    
    # Verify numerical equivalence
    if Q_original is not None:
        max_diff = torch.max(torch.abs(Q_original - Q_optimized)).item()
        print(f"Maximum difference between original and optimized: {max_diff:.2e}")
        if max_diff < 1e-5:
            print("✓ Results are numerically equivalent")
        else:
            print("✗ WARNING: Results differ significantly!")
        print()
    
    # Report speedup
    if avg_original is not None:
        speedup = avg_original / avg_optimized
        print(f"SPEEDUP: {speedup:.1f}×")
    else:
        print(f"EXECUTION TIME: {avg_optimized:.4f} seconds")
    print()
    
    return Q_optimized


def main():
    """Run benchmarks for different dimensions."""
    print("\n" + "="*70)
    print("TENSOR PRODUCT OPTIMIZATION BENCHMARK")
    print("="*70)
    
    # Check for GPU
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("! No GPU available, using CPU")
    
    # Test different dimensions
    dimensions = [5, 8, 10]
    
    for d in dimensions:
        try:
            benchmark_tensor_product(d, num_runs=3, device=device)
        except Exception as e:
            print(f"✗ Error at d={d}: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
