"""
High-dimensional two-layer algorithm combining projected subgradient with distorted greedy.
Uses vectorized tensor-product ops from psg_hd for d up to ~10.
"""
import torch
import os
import sys
from typing import Tuple

# Ensure psg_hd importable
sys.path.append(os.path.join(os.path.dirname(__file__), '../../psg_hd'))

from psg_hd.src.algorithms.projected_subgradient_optimized import (
    compute_subgradient as subgrad_h_opt,
)
from ..utils.objectives_hd import calculate_f, calculate_g, calculate_c
from two_layer.utils.projection import project_to_simplex
from .distorted_greedy_hd import distorted_greedy_step


def projected_subgradient_inner_loop(
    S: list[set], w_init: torch.Tensor,
    P_list, pi_list,
    d: int, states: list, state_to_idx: dict,
    K: int, eta: float
) -> Tuple[torch.Tensor, list[torch.Tensor], list[float]]:
    """
    Run K steps of projected subgradient ascent on f(S, w) (equivalently descent on h).
    """
    w = w_init.clone()
    w_hist, inner_f_vals = [], []

    for _ in range(K):
        # Build partition S plus complement
        supp_S = set().union(*S) if len(S) > 0 else set()
        complement = sorted(list(set(range(d)) - supp_S))
        partition = [sorted(list(s)) for s in S if len(s) > 0]
        if len(complement) > 0:
            partition.append(complement)

        g = subgrad_h_opt(w, P_list, pi_list, partition, d, states, state_to_idx, recompute_pi_bar=True)
        v = w - eta * g
        w = project_to_simplex(v)
        w_hist.append(w.clone())
        inner_f_vals.append(float(calculate_f(S, w, P_list, pi_list, d, states, state_to_idx).item()))

    w_avg = torch.mean(torch.stack(w_hist), dim=0)
    return w_avg, w_hist, inner_f_vals


def two_layer_algorithm_hd(P_list, pi_list, d: int, states: list, state_to_idx: dict,
                           V: list[set], l: int, K: int, eta: float,
                           beta: float = 0.0, verbose: bool = True,
                           return_history: bool = False):
    """
    High-d two-layer algorithm (Algorithm 2):
      max_{S ⪯ V, |supp(S)| ≤ l} f(S, w*(S)), with w*(S) ≈ PSG average.
    """
    device = P_list[0].device
    n = len(P_list)

    S = [set() for _ in range(len(V))]
    w = torch.full((n,), 1.0 / n, device=device)

    history = {
        'f_values': [], 'g_values': [], 'c_values': [],
        'S_before': [], 'S_after': [],
        'w_before_psg': [], 'w_after_psg': [],
        'inner_f_values': [], 'selected_elements': [], 'gains': []
    }

    if verbose:
        print("Two-Layer Algorithm (HD)")
        print("======================")
        print(f"n models: {n}\nDimension d: {d}\n|V|: {len(V)} (m = {len(V)+1})")
        print(f"l (cardinality): {l}\nK (inner iters): {K}\neta: {eta:.6f}")
        print(f"V: {V}\n")

    for i in range(l):
        if verbose:
            print(f"Iteration {i+1}/{l}")
            print(f"  S: {S} (|supp|={sum(len(s) for s in S)})")

        history['S_before'].append([s.copy() for s in S])
        history['w_before_psg'].append(w.clone())

        w_avg, w_inner_hist, inner_f = projected_subgradient_inner_loop(
            S, w.clone(), P_list, pi_list, d, states, state_to_idx, K, eta
        )
        w = w_avg
        history['w_after_psg'].append(w.clone())

        if verbose:
            f_cur = calculate_f(S, w, P_list, pi_list, d, states, state_to_idx)
            print(f"  After PSG: f(S, w̄) = {f_cur.item():.6f}")

        S_new, best_j, best_e, best_gain = distorted_greedy_step(
            S, w, P_list, pi_list, d, states, state_to_idx, V, l, i, beta
        )

        if verbose:
            if best_gain > 0:
                print(f"  Picked j={best_j}, e={best_e}, gain={best_gain:.6f}")
            else:
                print(f"  No positive gain (best={best_gain:.6f})")

        S = S_new
        history['S_after'].append([s.copy() for s in S])

        if return_history:
            history['inner_f_values'].append(inner_f)
            history['selected_elements'].append((best_j, best_e))
            history['gains'].append(best_gain)
            # Track outer f/g/c with updated S
            f_val = calculate_f(S, w, P_list, pi_list, d, states, state_to_idx)
            g_val = calculate_g(S, w, P_list, pi_list, d, states, state_to_idx, V, beta)
            c_val = calculate_c(S, w, P_list, pi_list, d, states, state_to_idx, V, beta)
            history['f_values'].append(f_val.item())
            history['g_values'].append(g_val.item())
            history['c_values'].append(c_val.item())

        if verbose:
            print()

    f_final = calculate_f(S, w, P_list, pi_list, d, states, state_to_idx)
    if verbose:
        print("Final Results (HD)")
        print("=================")
        print(f"S: {S}\n|supp(S)|={sum(len(s) for s in S)}")
        print(f"w: {w}")
        print(f"f(S, w): {f_final.item():.6f}")

    if return_history:
        return S, w, f_final, history
    return S, w, f_final

