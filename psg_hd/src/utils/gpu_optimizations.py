"""
GPU optimization utilities for high-dimensional Markov chain experiments.
Implements parallel computing strategies for d=14 experiments.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
import math

def batch_matrix_power(P: torch.Tensor, powers: List[int]) -> torch.Tensor:
    """
    Efficiently compute multiple powers of a matrix P in parallel.
    
    Args:
        P: Base transition matrix (n_states x n_states)
        powers: List of powers to compute [1, 2, 4, 8, 16]
    
    Returns:
        Tensor of shape (len(powers), n_states, n_states) containing P^k for each k in powers
    """
    device = P.device
    n_states = P.shape[0]
    n_powers = len(powers)
    
    # Pre-allocate result tensor
    result = torch.zeros(n_powers, n_states, n_states, device=device, dtype=P.dtype)
    
    # Store intermediate powers for efficient computation
    power_cache = {1: P}
    
    for i, power in enumerate(powers):
        if power in power_cache:
            result[i] = power_cache[power]
        else:
            # Find the most efficient way to compute this power
            if power == 2:
                result[i] = torch.matmul(P, P)
            elif power % 2 == 0:
                half_power = power // 2
                if half_power in power_cache:
                    half_mat = power_cache[half_power]
                else:
                    # Compute recursively
                    half_mat = matrix_power_efficient(P, half_power)
                    power_cache[half_power] = half_mat
                result[i] = torch.matmul(half_mat, half_mat)
            else:
                # Odd power: P^k = P * P^(k-1)
                prev_power = power - 1
                if prev_power in power_cache:
                    prev_mat = power_cache[prev_power]
                else:
                    prev_mat = matrix_power_efficient(P, prev_power)
                    power_cache[prev_power] = prev_mat
                result[i] = torch.matmul(P, prev_mat)
            
            power_cache[power] = result[i]
    
    return result

def matrix_power_efficient(P: torch.Tensor, k: int) -> torch.Tensor:
    """Compute P^k using binary exponentiation for efficiency."""
    if k == 0:
        return torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
    if k == 1:
        return P
    
    if k % 2 == 0:
        half = matrix_power_efficient(P, k // 2)
        return torch.matmul(half, half)
    else:
        return torch.matmul(P, matrix_power_efficient(P, k - 1))

def batch_kl_divergence(P_batch: torch.Tensor, Q_batch: torch.Tensor, pi_batch: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence for multiple matrix pairs in parallel.
    
    Args:
        P_batch: Tensor of shape (batch_size, n_states, n_states)
        Q_batch: Tensor of shape (batch_size, n_states, n_states) 
        pi_batch: Tensor of shape (batch_size, n_states)
    
    Returns:
        Tensor of shape (batch_size,) containing KL divergences
    """
    eps = 1e-12
    
    # Clamp to avoid log(0)
    P_clamped = torch.clamp(P_batch, min=eps)
    Q_clamped = torch.clamp(Q_batch, min=eps)
    
    # Compute log ratio: log(P/Q)
    log_ratio = torch.log(P_clamped) - torch.log(Q_clamped)
    
    # Weighted sum: π(x) * Σ_y P(x,y) * log(P(x,y)/Q(x,y))
    # Shape: (batch_size, n_states)
    weighted_sum = torch.sum(P_batch * log_ratio, dim=2)
    
    # Final sum over states: Σ_x π(x) * weighted_sum(x)
    # Shape: (batch_size,)
    kl_divs = torch.sum(pi_batch * weighted_sum, dim=1)
    
    return kl_divs

def parallel_marginalization(P: torch.Tensor, pi: torch.Tensor, partitions: List[List[int]], 
                           d: int, states: List, state_to_idx: dict) -> List[torch.Tensor]:
    """
    Compute marginalizations for multiple partitions in parallel.
    
    Args:
        P: Transition matrix
        pi: Stationary distribution  
        partitions: List of coordinate subsets to marginalize over
        d: Total dimension
        states: List of all states
        state_to_idx: State to index mapping
    
    Returns:
        List of marginalized matrices
    """
    device = P.device
    results = []
    
    # Group partitions by size for more efficient batch processing
    partition_groups = {}
    for i, partition in enumerate(partitions):
        size = len(partition)
        if size not in partition_groups:
            partition_groups[size] = []
        partition_groups[size].append((i, partition))
    
    # Process each group
    for size, group in partition_groups.items():
        if size == 0:
            # Empty partition case
            for i, _ in group:
                results.append(torch.ones((1, 1), device=device))
            continue
            
        # Batch process partitions of the same size
        batch_results = []
        for i, partition in group:
            # Use existing marginalization logic but optimize it
            marginal_matrix = marginalize_optimized(P, pi, partition, d, states, state_to_idx)
            batch_results.append(marginal_matrix)
        
        results.extend(batch_results)
    
    return results

def marginalize_optimized(P: torch.Tensor, pi: torch.Tensor, S: List[int], 
                         d: int, states: List, state_to_idx: dict) -> torch.Tensor:
    """
    Optimized marginalization using advanced indexing and vectorized operations.
    """
    if not S:
        return torch.ones((1, 1), device=P.device)
    
    device = P.device
    n_states = len(states)
    
    # Create marginalization mapping more efficiently
    marginal_map = {}
    marginal_states = []
    
    for state in states:
        marginal_state = tuple(state[j] for j in S)
        if marginal_state not in marginal_map:
            marginal_map[marginal_state] = len(marginal_states)
            marginal_states.append(marginal_state)
    
    n_marginal = len(marginal_states)
    
    # Create index mapping tensors for vectorized operations
    state_to_marginal = torch.zeros(n_states, dtype=torch.long, device=device)
    for i, state in enumerate(states):
        marginal_state = tuple(state[j] for j in S)
        state_to_marginal[i] = marginal_map[marginal_state]
    
    # Vectorized computation of marginal distribution
    pi_marginal = torch.zeros(n_marginal, device=device)
    pi_marginal.scatter_add_(0, state_to_marginal, pi)
    
    # Vectorized computation of transition probabilities
    P_marginal = torch.zeros(n_marginal, n_marginal, device=device)
    
    # This is the most compute-intensive part - optimize with advanced indexing
    for i in range(n_marginal):
        mask_i = (state_to_marginal == i)
        if pi_marginal[i] > 0:
            # Get transition probabilities from states in group i
            P_from_i = P[mask_i, :]  # Shape: (n_states_in_i, n_total_states)
            pi_from_i = pi[mask_i]   # Shape: (n_states_in_i,)
            
            for j in range(n_marginal):
                mask_j = (state_to_marginal == j)
                # Sum transitions to states in group j
                prob_to_j = P_from_i[:, mask_j].sum(dim=1)  # Shape: (n_states_in_i,)
                # Weighted by stationary probabilities
                weighted_prob = (pi_from_i * prob_to_j).sum()
                P_marginal[i, j] = weighted_prob / pi_marginal[i]
    
    return P_marginal

def memory_efficient_tensor_product(P_marginals: List[torch.Tensor], 
                                   original_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Compute tensor product of marginal matrices in a memory-efficient way.
    Uses iterative Kronecker products with memory management.
    """
    if not P_marginals:
        return torch.eye(original_shape[0], device=P_marginals[0].device)
    
    # Start with first marginal
    result = P_marginals[0]
    
    # Iteratively compute Kronecker products
    for P_marginal in P_marginals[1:]:
        # Clear GPU cache before large operation
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        result = torch.kron(result, P_marginal)
        
        # Monitor memory usage
        if hasattr(torch.cuda, 'memory_allocated'):
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            if memory_gb > 20:  # Close to A10G limit
                print(f"Warning: GPU memory usage: {memory_gb:.2f} GB")
    
    return result

def optimized_subgradient_computation(w: torch.Tensor, P_list: List[torch.Tensor], 
                                    pi_list: List[torch.Tensor], partition: List[List[int]],
                                    d: int, states: List, state_to_idx: dict) -> torch.Tensor:
    """
    Compute subgradient using optimized parallel operations.
    """
    n = len(P_list)
    device = P_list[0].device
    
    # Convert to batch tensors for parallel processing
    P_batch = torch.stack(P_list, dim=0)  # Shape: (n, n_states, n_states)
    pi_batch = torch.stack(pi_list, dim=0)  # Shape: (n, n_states)
    
    # Compute P_bar efficiently
    P_bar = torch.sum(w.unsqueeze(1).unsqueeze(2) * P_batch, dim=0)
    
    # Compute stationary distribution of P_bar
    pi_bar = stationary_distribution_optimized(P_bar)
    
    # Compute tensor product (this is still the bottleneck for d=14)
    P_bar_product = compute_tensor_product_optimized(P_bar, pi_bar, partition, d, states, state_to_idx)
    
    # Batch compute KL divergences
    P_bar_batch = P_bar_product.unsqueeze(0).expand(n, -1, -1)
    kl_divs = batch_kl_divergence(P_batch, P_bar_batch, pi_batch)
    
    # Subgradient computation
    kl_n = kl_divs[-1]  # KL divergence for last matrix
    subgrad = kl_n - kl_divs
    
    return subgrad

def stationary_distribution_optimized(P: torch.Tensor, max_iter: int = 10000, 
                                    tol: float = 1e-10) -> torch.Tensor:
    """
    Optimized stationary distribution computation using GPU acceleration.
    """
    n = P.shape[0]
    device = P.device
    v = torch.ones(n, device=device, dtype=P.dtype) / n
    
    for _ in range(max_iter):
        v_next = v @ P
        v_next = v_next / v_next.sum()
        
        if torch.norm(v_next - v, p=1) < tol:
            return v_next
        v = v_next
    
    return v

def compute_tensor_product_optimized(P: torch.Tensor, pi: torch.Tensor, 
                                   partition: List[List[int]], d: int, 
                                   states: List, state_to_idx: dict) -> torch.Tensor:
    """
    Memory-optimized tensor product computation for high dimensions.
    """
    device = P.device
    
    # Compute marginals in parallel where possible
    marginals = []
    for S in partition:
        P_S = marginalize_optimized(P, pi, S, d, states, state_to_idx)
        marginals.append(P_S)
    
    # Compute tensor product with memory management
    result = memory_efficient_tensor_product(marginals, P.shape)
    
    return result