"""
Objective functions for the two-layer optimization problem.
Implements f(S, w), g(S, w), c(S, w) following the manuscript.

Notes:
- We compute the factorized/product matrix on the ORIGINAL state space via
  projected_subgradient.algorithms.compute_tensor_product to match prior code.
- g(S, w) and c(S, w) implement equations analogous to (20) and (21):
  f = g - c, where g is (m-1)-submodular and c is non-negative modular (with
  a suitable choice of β). This file uses a user-provided β (default 0).
"""
import torch
import sys
import os

# Ensure projected_subgradient is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '../../projected_subgradient'))

from projected_subgradient.src.utils.matrix_ops import kl_divergence
from projected_subgradient.src.algorithms.projected_subgradient import (
    compute_tensor_product,
    stationary_distribution,
)

def _normalize_rows(M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-normalize a nonnegative matrix to make it row-stochastic."""
    denom = M.sum(dim=1, keepdim=True).clamp_min(eps)
    return M / denom
def compute_weighted_average_P(P_list: list[torch.Tensor], w: torch.Tensor) -> torch.Tensor:
    """
    Compute the weighted average transition matrix.
    
    Args:
        P_list: List of transition matrices [P_1, ..., P_n]
        w: Weight vector of shape (n,)
        
    Returns:
        Weighted average P_bar = sum_i w_i * P_i
    """
    P_bar = torch.zeros_like(P_list[0])
    for i, P in enumerate(P_list):
        P_bar += w[i] * P
    return P_bar


def compute_product_matrix(
    P_bar: torch.Tensor,
    partition_S: list[list[int]],
    d: int,
    states: list,
    state_to_idx: dict,
) -> torch.Tensor:
    """
    Compute the product matrix on the ORIGINAL state space:
        Q = ⊗_{j} P_bar^(S_j)
    using the same construction as in projected_subgradient.

    Args:
        P_bar: Weighted average transition matrix on full state space
        partition_S: Partition as list of lists (each S_j a list of coords)
        d: Dimension
        states: All states of the original Markov chain
        state_to_idx: Mapping from state to index

    Returns:
        Q: Tensor-product-like matrix on the original state space
    """
    # Recompute stationary distribution of P_bar for correct marginalization
    pi_bar = stationary_distribution(P_bar)
    Q = compute_tensor_product(P_bar, pi_bar, partition_S, d, states, state_to_idx)
    # Ensure Q is a valid transition matrix (row-stochastic)
    Q = _normalize_rows(Q)
    return Q


def calculate_f(
    S: list[set],
    w: torch.Tensor,
    P_list: list[torch.Tensor],
    pi_list: list[torch.Tensor],
    d: int,
    states: list,
    state_to_idx: dict,
) -> torch.Tensor:
    """
    Calculate f(S, w) = sum_i w_i * D_KL^pi(P_i || Q),
    where Q is computed on the original state space via compute_tensor_product.
    
    This is equation (17) from the manuscript.
    
    Args:
        S: List of sets [S_1, ..., S_{m-1}]
        w: Weight vector
        P_list: List of transition matrices
        pi_list: List of stationary distributions for each P_i
        d: Dimension of state space
        states: All states (original space)
        state_to_idx: Mapping from state to index
        
    Returns:
        Scalar value f(S, w)
    """
    # Compute weighted average
    P_bar = compute_weighted_average_P(P_list, w)

    # Build full partition including complement to cover all coordinates
    S_lists = [sorted(list(s)) for s in S if len(s) > 0]
    supp_S = set().union(*S) if len(S) > 0 else set()
    complement = sorted(list(set(range(d)) - supp_S))
    partition = list(S_lists)
    if len(complement) > 0:
        partition.append(complement)
    
    # If partition is empty (no S and complement empty), fallback to P_bar
    if len(partition) == 0:
        Q = P_bar
    else:
        Q = compute_product_matrix(P_bar, partition, d, states, state_to_idx)
    
    # Compute weighted KL divergence
    f_value = torch.tensor(0.0, device=w.device)
    for i, P in enumerate(P_list):
        f_value += w[i] * kl_divergence(P, Q, pi_list[i])
    
    return f_value


def calculate_c(
    S: list[set],
    w: torch.Tensor,
    P_list: list[torch.Tensor],
    pi_list: list[torch.Tensor],
    d: int,
    states: list,
    state_to_idx: dict,
    V: list[set],
    beta: float = 0.0,
) -> torch.Tensor:
    """
    Calculate c(S, w) = -beta + sum_{j=1}^{m-1} sum_{e in S_j} [marginal gain term].
    
    This is the modular function from equation (21).
    
    Args:
        S: List of sets [S_1, ..., S_{m-1}]
        w: Weight vector
        P_list: List of transition matrices
        pi_list: List of stationary distributions
        d: Dimension of state space
        states: All states
        state_to_idx: State mapping
        V: Full partition [V_1, ..., V_{m-1}]
        beta: Constant (default 0)
        
    Returns:
        Scalar value c(S, w)
    """
    P_bar = compute_weighted_average_P(P_list, w)
    pi_bar = stationary_distribution(P_bar)
    
    c_value = torch.tensor(-beta, device=w.device, dtype=torch.float32)
    
    # Compute support of V
    supp_V = set()
    for V_j in V:
        supp_V = supp_V.union(V_j)
    
    for j in range(len(S)):
        for e in S[j]:
            if j >= len(V):
                continue  # Skip if S has more sets than V
            
            # First term: D_KL(P_bar^{(V_j)} || P_bar^{(V_j \ {e})} ⊗ P_bar^{(e)})
            V_j_partition = [sorted(list(V[j]))]
            Q_Vj = compute_product_matrix(P_bar, V_j_partition, d, states, state_to_idx)
            
            V_j_minus_e = V[j] - {e}
            if len(V_j_minus_e) > 0:
                V_j_minus_e_partition = [sorted(list(V_j_minus_e)), [e]]
            else:
                V_j_minus_e_partition = [[e]]
            Q_factorized = compute_product_matrix(P_bar, V_j_minus_e_partition, d, states, state_to_idx)
            
            term1 = kl_divergence(Q_Vj, Q_factorized, pi_bar)
            
            # Second term: D_KL(P_bar^{(-supp(V) \ {e})} || P_bar^{(-supp(V))} ⊗ P_bar^{(e)})
            ground_set = set(range(d))
            complement_V = ground_set - supp_V
            
            if len(complement_V) > 0:
                complement_minus_e = complement_V - {e} if e in complement_V else complement_V
                if len(complement_minus_e) > 0:
                    comp_partition = [sorted(list(complement_minus_e))]
                    Q_comp = compute_product_matrix(P_bar, comp_partition, d, states, state_to_idx)
                    
                    comp_factorized_partition = [sorted(list(complement_V)), [e]]
                    Q_comp_fact = compute_product_matrix(P_bar, comp_factorized_partition, d, states, state_to_idx)
                    
                    term2 = kl_divergence(Q_comp, Q_comp_fact, pi_bar)
                    c_value += term1 - term2
                else:
                    c_value += term1
            else:
                c_value += term1
    
    return c_value


def calculate_g(
    S: list[set],
    w: torch.Tensor,
    P_list: list[torch.Tensor],
    pi_list: list[torch.Tensor],
    d: int,
    states: list,
    state_to_idx: dict,
    V: list[set],
    beta: float = 0.0,
) -> torch.Tensor:
    """
    Calculate g(S, w) = f(S, w) - beta + [modular correction term].
    
    This is the (m-1)-submodular function from equation (20).
    
    Args:
        S: List of sets [S_1, ..., S_{m-1}]
        w: Weight vector
        P_list: List of transition matrices
        pi_list: List of stationary distributions
        d: Dimension of state space
        states: All states
        state_to_idx: State mapping
        V: Full partition [V_1, ..., V_{m-1}]
        beta: Constant (default 0)
        
    Returns:
        Scalar value g(S, w)
    """
    f_value = calculate_f(S, w, P_list, pi_list, d, states, state_to_idx)
    P_bar = compute_weighted_average_P(P_list, w)
    pi_bar = stationary_distribution(P_bar)
    
    modular_term = torch.tensor(0.0, device=w.device, dtype=torch.float32)
    
    # Support of V and its complement
    supp_V = set().union(*V) if len(V) > 0 else set()
    ground_set = set(range(d))
    complement_V = ground_set - supp_V
    
    # Compute the modular correction term over elements PRESENT IN S (not whole V)
    for j in range(len(S)):
        for e in S[j]:
            # First part: D( P̄^{(V_j)} || P̄^{(V_j\{e})} ⊗ P̄^{(e)} )
            V_j_partition = [sorted(list(V[j]))]
            Q_Vj = compute_product_matrix(P_bar, V_j_partition, d, states, state_to_idx)
            
            V_j_minus_e = V[j] - {e}
            if len(V_j_minus_e) > 0:
                V_j_minus_e_partition = [sorted(list(V_j_minus_e)), [e]]
            else:
                V_j_minus_e_partition = [[e]]
            Q_factorized = compute_product_matrix(P_bar, V_j_minus_e_partition, d, states, state_to_idx)
            
            term_A = kl_divergence(Q_Vj, Q_factorized, pi_bar)
            
            # Second part: D( P̄^{(-supp(V)\{e})} || P̄^{(-supp(V))} ⊗ P̄^{(e)} )
            term_B = torch.tensor(0.0, device=w.device, dtype=torch.float32)
            if len(complement_V) > 0:
                comp_minus_e = complement_V - {e} if e in complement_V else complement_V
                if len(comp_minus_e) > 0:
                    comp_partition = [sorted(list(comp_minus_e))]
                    Q_comp = compute_product_matrix(P_bar, comp_partition, d, states, state_to_idx)
                    comp_factorized_partition = [sorted(list(complement_V)), [e]]
                    Q_comp_fact = compute_product_matrix(P_bar, comp_factorized_partition, d, states, state_to_idx)
                    term_B = kl_divergence(Q_comp, Q_comp_fact, pi_bar)
            
            modular_term += (term_A - term_B)
    
    return f_value - beta + modular_term


def compute_subgradient_placeholder(*args, **kwargs):
    """
    Deprecated local subgradient (kept for backward compatibility).
    Use projected_subgradient.src.algorithms.projected_subgradient.compute_subgradient
    directly in the two-layer algorithm to ensure consistency with h(w).
    """
    raise NotImplementedError("Use projected_subgradient.compute_subgradient in two-layer algorithm.")
