"""
Vectorized objective functions for the high-d two-layer optimization problem.

Implements f(S, w), g(S, w), c(S, w) using psg_hd's vectorized tensor-product
operator on the ORIGINAL state space (no Kronecker lift).
"""
import torch
import os
import sys

# Ensure psg_hd is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '../../psg_hd'))

from psg_hd.src.algorithms.projected_subgradient_optimized import (
    compute_tensor_product_vectorized,
    stationary_distribution,
)
from projected_subgradient.src.utils.matrix_ops import kl_divergence


def _normalize_rows(M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = M.sum(dim=1, keepdim=True).clamp_min(eps)
    return M / denom


def compute_weighted_average_P(P_list: list[torch.Tensor], w: torch.Tensor) -> torch.Tensor:
    P_bar = torch.zeros_like(P_list[0])
    for i, P in enumerate(P_list):
        P_bar += w[i] * P
    return P_bar


def compute_product_matrix(P_bar: torch.Tensor,
                           partition_S: list[list[int]],
                           d: int,
                           states: list,
                           state_to_idx: dict) -> torch.Tensor:
    """
    Compute Q = ⊗_j P_bar^(S_j) on the ORIGINAL state space using vectorized ops.
    """
    if not partition_S or len(partition_S) == 0:
        return P_bar
    pi_bar = stationary_distribution(P_bar)
    Q = compute_tensor_product_vectorized(P_bar, pi_bar, partition_S, d, states, state_to_idx)
    Q = _normalize_rows(Q)
    return Q


def calculate_f(S: list[set], w: torch.Tensor,
                P_list: list[torch.Tensor], pi_list: list[torch.Tensor],
                d: int, states: list, state_to_idx: dict) -> torch.Tensor:
    """
    f(S, w) = Σ_i w_i · D_KL(P_i || Q), with Q = (⊗_{j=1}^{m-1} P̄^(S_j)) ⊗ P̄^(-supp(S)).
    """
    P_bar = compute_weighted_average_P(P_list, w)

    # Build partition including complement of supp(S)
    S_lists = [sorted(list(s)) for s in S if len(s) > 0]
    supp_S = set().union(*S) if len(S) > 0 else set()
    complement = sorted(list(set(range(d)) - supp_S))
    partition = list(S_lists)
    if len(complement) > 0:
        partition.append(complement)

    Q = compute_product_matrix(P_bar, partition, d, states, state_to_idx)

    f_value = torch.tensor(0.0, device=w.device)
    for i, P in enumerate(P_list):
        f_value += w[i] * kl_divergence(P, Q, pi_list[i])
    return f_value


def calculate_c(S: list[set], w: torch.Tensor,
                P_list: list[torch.Tensor], pi_list: list[torch.Tensor],
                d: int, states: list, state_to_idx: dict,
                V: list[set], beta: float = 0.0) -> torch.Tensor:
    """
    c(S, w) = -β + Σ_{j=1}^{m-1} Σ_{e∈S_j} [ D(P̄^{(V_j)} || P̄^{(V_j\{e})} ⊗ P̄^{(e)})
                                             - D(P̄^{(-supp(V)\{e})} || P̄^{(-supp(V))} ⊗ P̄^{(e)}) ].
    """
    P_bar = compute_weighted_average_P(P_list, w)
    pi_bar = stationary_distribution(P_bar)

    c_value = torch.tensor(-beta, device=w.device, dtype=torch.float32)

    # Support of V and its complement
    supp_V = set().union(*V) if len(V) > 0 else set()
    ground = set(range(d))
    complement_V = ground - supp_V

    for j in range(len(S)):
        for e in S[j]:
            # Term 1
            Vj_part = [sorted(list(V[j]))]
            Q_Vj = compute_product_matrix(P_bar, Vj_part, d, states, state_to_idx)

            Vj_minus_e = V[j] - {e}
            if len(Vj_minus_e) > 0:
                fact_part = [sorted(list(Vj_minus_e)), [e]]
            else:
                fact_part = [[e]]
            Q_fact = compute_product_matrix(P_bar, fact_part, d, states, state_to_idx)
            term1 = kl_divergence(Q_Vj, Q_fact, pi_bar)

            # Term 2
            term2 = torch.tensor(0.0, device=w.device, dtype=torch.float32)
            if len(complement_V) > 0:
                comp_minus_e = complement_V - {e} if e in complement_V else complement_V
                if len(comp_minus_e) > 0:
                    comp_part = [sorted(list(comp_minus_e))]
                    Q_comp = compute_product_matrix(P_bar, comp_part, d, states, state_to_idx)
                    comp_fact_part = [sorted(list(complement_V)), [e]]
                    Q_comp_fact = compute_product_matrix(P_bar, comp_fact_part, d, states, state_to_idx)
                    term2 = kl_divergence(Q_comp, Q_comp_fact, pi_bar)

            c_value += term1 - term2

    return c_value


def calculate_g(S: list[set], w: torch.Tensor,
                P_list: list[torch.Tensor], pi_list: list[torch.Tensor],
                d: int, states: list, state_to_idx: dict,
                V: list[set], beta: float = 0.0) -> torch.Tensor:
    """
    g(S, w) = f(S, w) - β + Σ_{j,e∈S_j}[ D(P̄^{(V_j)} || P̄^{(V_j\{e})} ⊗ P̄^{(e)})
                                          - D(P̄^{(-supp(V)\{e})} || P̄^{(-supp(V))} ⊗ P̄^{(e)}) ].
    Ensures f = g - c.
    """
    f_val = calculate_f(S, w, P_list, pi_list, d, states, state_to_idx)

    P_bar = compute_weighted_average_P(P_list, w)
    pi_bar = stationary_distribution(P_bar)

    supp_V = set().union(*V) if len(V) > 0 else set()
    ground = set(range(d))
    complement_V = ground - supp_V

    modular = torch.tensor(0.0, device=w.device, dtype=torch.float32)
    for j in range(len(S)):
        for e in S[j]:
            # Term A
            Vj_part = [sorted(list(V[j]))]
            Q_Vj = compute_product_matrix(P_bar, Vj_part, d, states, state_to_idx)

            Vj_minus_e = V[j] - {e}
            if len(Vj_minus_e) > 0:
                fact_part = [sorted(list(Vj_minus_e)), [e]]
            else:
                fact_part = [[e]]
            Q_fact = compute_product_matrix(P_bar, fact_part, d, states, state_to_idx)
            termA = kl_divergence(Q_Vj, Q_fact, pi_bar)

            # Term B
            termB = torch.tensor(0.0, device=w.device, dtype=torch.float32)
            if len(complement_V) > 0:
                comp_minus_e = complement_V - {e} if e in complement_V else complement_V
                if len(comp_minus_e) > 0:
                    comp_part = [sorted(list(comp_minus_e))]
                    Q_comp = compute_product_matrix(P_bar, comp_part, d, states, state_to_idx)
                    comp_fact_part = [sorted(list(complement_V)), [e]]
                    Q_comp_fact = compute_product_matrix(P_bar, comp_fact_part, d, states, state_to_idx)
                    termB = kl_divergence(Q_comp, Q_comp_fact, pi_bar)

            modular += (termA - termB)

    return f_val - beta + modular

