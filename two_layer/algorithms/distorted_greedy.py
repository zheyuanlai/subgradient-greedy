"""
Distorted greedy algorithm for submodular optimization.
Implements the outer loop of Algorithm 2.
"""
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../projected_subgradient'))

from ..utils.objectives import calculate_g, calculate_c


def compute_distorted_marginal_gain(
    S: list[set], e: int, j: int, w: torch.Tensor,
    P_list: list[torch.Tensor], pi_list: list[torch.Tensor],
    d: int, states: list, state_to_idx: dict,
    V: list[set], l: int, i: int, beta: float = 0.0
) -> torch.Tensor:
    """
    Compute the distorted marginal gain for adding element e to set S_j.
    
    Distorted gain = (1 - 1/l)^(l - (i+1)) * Δ_{e,j} g(S, w) - c({e}, w)
    
    Args:
        S: Current partition [S_1, ..., S_{m-1}]
        e: Element to add
        j: Index of the set to add element to
        w: Weight vector
        P_list: List of transition matrices
        pi: Stationary distribution
        d: Dimension of state space
        V: Full partition [V_1, ..., V_{m-1}]
        l: Cardinality constraint
        i: Current iteration (0-indexed)
        beta: Constant for g and c
        
    Returns:
        Distorted marginal gain value
    """
    # Compute g(S, w)
    g_S = calculate_g(S, w, P_list, pi_list, d, states, state_to_idx, V, beta)
    
    # Compute g(S with e added to S_j, w)
    S_plus_e = [s.copy() for s in S]
    S_plus_e[j].add(e)
    g_S_plus_e = calculate_g(S_plus_e, w, P_list, pi_list, d, states, state_to_idx, V, beta)
    
    # Marginal gain
    delta_g = g_S_plus_e - g_S
    
    # Compute c({e}, w) - the cost of adding just element e
    S_singleton = [set() for _ in range(len(S))]
    S_singleton[j] = {e}
    c_e = calculate_c(S_singleton, w, P_list, pi_list, d, states, state_to_idx, V, beta)
    
    # Distortion factor
    distortion = (1 - 1/l) ** (l - (i + 1))
    
    # Distorted marginal gain
    distorted_gain = distortion * delta_g - c_e
    
    return distorted_gain


def distorted_greedy_step(
    S: list[set], w: torch.Tensor, P_list: list[torch.Tensor], pi_list: list[torch.Tensor],
    d: int, states: list, state_to_idx: dict, V: list[set], l: int, i: int,
    beta: float = 0.0
) -> tuple[list[set], int, int, float]:
    """
    Perform one step of the distorted greedy algorithm.
    
    Finds the best element to add and updates S.
    
    Args:
        S: Current partition [S_1, ..., S_{m-1}]
        w: Weight vector
        P_list: List of transition matrices
        pi: Stationary distribution
        d: Dimension of state space
        V: Full partition [V_1, ..., V_{m-1}]
        l: Cardinality constraint
        i: Current iteration (0-indexed)
        beta: Constant for g and c
        
    Returns:
        Tuple of (updated S, best j, best e, best gain)
    """
    best_gain = torch.tensor(-float('inf'), device=w.device)
    best_j = -1
    best_e = -1
    
    # Iterate over all partitions
    for j in range(len(V)):
        # Iterate over all elements in V_j that are not yet in S_j
        for e in V[j]:
            if e not in S[j]:
                # Compute distorted marginal gain
                gain = compute_distorted_marginal_gain(S, e, j, w, P_list, pi_list, d, states, state_to_idx, V, l, i, beta)
                
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
                    best_e = e
    
    # Update S if the gain is positive
    S_new = [s.copy() for s in S]
    if best_gain > 0:
        S_new[best_j].add(best_e)
    
    return S_new, best_j, best_e, best_gain.item()
