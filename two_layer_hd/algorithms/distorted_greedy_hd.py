"""
Distorted greedy step for the high-d two-layer algorithm (vectorized backend).
"""
import torch
from ..utils.objectives_hd import calculate_g, calculate_c


def compute_distorted_marginal_gain(S: list[set], e: int, j: int, w: torch.Tensor,
                                    P_list, pi_list, d, states, state_to_idx,
                                    V: list[set], l: int, i: int, beta: float = 0.0) -> torch.Tensor:
    """
    Distorted gain = (1 - 1/l)^(l - (i+1)) * Δ_{e,j} g(S, w) - c({e}, w)
    """
    g_S = calculate_g(S, w, P_list, pi_list, d, states, state_to_idx, V, beta)

    S_plus = [s.copy() for s in S]
    S_plus[j].add(e)
    g_S_plus = calculate_g(S_plus, w, P_list, pi_list, d, states, state_to_idx, V, beta)
    delta_g = g_S_plus - g_S

    S_singleton = [set() for _ in range(len(S))]
    S_singleton[j] = {e}
    c_e = calculate_c(S_singleton, w, P_list, pi_list, d, states, state_to_idx, V, beta)

    distortion = (1 - 1 / l) ** (l - (i + 1))
    return distortion * delta_g - c_e


def distorted_greedy_step(S: list[set], w: torch.Tensor,
                          P_list, pi_list, d, states, state_to_idx,
                          V: list[set], l: int, i: int, beta: float = 0.0):
    """
    One distorted greedy step: pick (j*, e*) with maximum distorted gain.
    """
    best_gain = torch.tensor(-float('inf'), device=w.device)
    best_j, best_e = -1, -1

    for j in range(len(V)):
        for e in V[j]:
            if e not in S[j]:
                gain = compute_distorted_marginal_gain(S, e, j, w, P_list, pi_list,
                                                       d, states, state_to_idx, V, l, i, beta)
                if gain > best_gain:
                    best_gain = gain
                    best_j, best_e = j, e

    S_new = [s.copy() for s in S]
    if best_gain > 0:
        S_new[best_j].add(best_e)
    return S_new, best_j, best_e, best_gain.item()

