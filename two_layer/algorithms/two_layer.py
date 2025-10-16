"""
Two-layer algorithm combining projected subgradient descent with distorted greedy.
Implements Algorithm 2 from the manuscript.
"""
import torch
import sys
import os
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '../../projected_subgradient'))

from ..utils.projection import project_to_simplex
from ..utils.objectives import (
    calculate_f,
    calculate_g,
    calculate_c,
)
from .distorted_greedy import distorted_greedy_step
from projected_subgradient.src.algorithms.projected_subgradient import compute_subgradient as subgrad_h


def projected_subgradient_inner_loop(
    S: list[set], w_init: torch.Tensor,
    P_list: list[torch.Tensor], pi_list: list[torch.Tensor],
    d: int, states: list, state_to_idx: dict,
    K: int, eta: float
) -> tuple[torch.Tensor, list[torch.Tensor], list[float]]:
    """
    Run K steps of projected subgradient ascent for fixed S.
    
    Maximizes f(S, w) over w in the simplex.
    
    Args:
        S: Fixed partition [S_1, ..., S_{m-1}]
        w_init: Initial weight vector
        P_list: List of transition matrices
        pi: Stationary distribution
        d: Dimension of state space
        K: Number of iterations
        eta: Step size
        
    Returns:
        Tuple of (averaged weights, list of all weight iterates)
    """
    w = w_init.clone()
    w_history = []
    inner_f_values: list[float] = []
    
    for k in range(K):
        # Subgradient of h(w) at current w with fixed partition S
        # Build full partition = S parts + complement to cover all coordinates
        partition = [sorted(list(s)) for s in S if len(s) > 0]
        supp_S = set().union(*S) if len(S) > 0 else set()
        complement = sorted(list(set(range(d)) - supp_S))
        if len(complement) > 0:
            partition.append(complement)
        g = subgrad_h(w, P_list, pi_list, partition, d, states, state_to_idx, recompute_pi_bar=True)
        # Descent step on h(w): v = w - eta * g (equivalently ascent on f)
        v = w - eta * g
        
        # Project onto simplex
        w = project_to_simplex(v)
        
        w_history.append(w.clone())
        # Track inner objective trajectory f(S, w^{(k)})
        f_k = calculate_f(S, w, P_list, pi_list, d, states, state_to_idx)
        inner_f_values.append(float(f_k.item()))
    
    # Average the weights
    w_avg = torch.mean(torch.stack(w_history), dim=0)
    
    return w_avg, w_history, inner_f_values


def two_layer_algorithm(P_list: list[torch.Tensor], pi_list: list[torch.Tensor], d: int,
                        states: list, state_to_idx: dict,
                        V: list[set], l: int, K: int, eta: float,
                        beta: float = 0.0, verbose: bool = True,
                        return_history: bool = False) -> tuple:
    """
    Solve the max-min-max optimization problem using the two-layer algorithm.
    
    Implements Algorithm 2 from the manuscript:
    max_{S ⪯ V, |supp(S)| ≤ l} f(S, w*(S))
    
    Args:
        P_list: List of n transition matrices
        pi: Stationary distribution
        d: Dimension of state space
        V: Full partition [V_1, ..., V_{m-1}]
        l: Cardinality constraint
        K: Number of inner loop iterations
        eta: Step size for subgradient descent
        beta: Constant for g and c functions
        verbose: Whether to print progress
        return_history: Whether to return full history
        
    Returns:
        If return_history=False: (S_final, w_final, f_final)
        If return_history=True: (S_final, w_final, f_final, history_dict)
    """
    device = P_list[0].device
    n = len(P_list)
    m = len(V) + 1  # Number of partitions (m-1 sets in V)
    
    # Initialize
    S = [set() for _ in range(len(V))]
    w = torch.full((n,), 1.0 / n, device=device)
    
    # History tracking
    history = {
        'f_values': [],
        'g_values': [],
        'c_values': [],
        'S_history': [],
        'w_history': [],
        'w_avg_history': [],
        'inner_f_values': [],
        'w_before_psg': [],
        'w_after_psg': [],
        'S_before': [],
        'S_after': [],
        'selected_elements': [],
        'gains': []
    }
    
    if verbose:
        print(f"Two-Layer Algorithm")
        print(f"==================")
        print(f"Number of models: {n}")
        print(f"Dimension: {d}")
        print(f"Partitions (m): {m}")
        print(f"Cardinality constraint (l): {l}")
        print(f"Inner iterations (K): {K}")
        print(f"Step size (eta): {eta}")
        print(f"Partition V: {V}")
        print()
    
    # Main loop: iterate l times to add l elements
    for i in range(l):
        if verbose:
            print(f"Iteration {i+1}/{l}")
            print(f"  Current S: {S}")
            print(f"  Current |supp(S)|: {sum(len(s) for s in S)}")
        
        # Inner loop: Projected subgradient descent
        # Record S and w at start of outer iteration i
        history['S_before'].append([s.copy() for s in S])
        history['w_before_psg'].append(w.clone())
        w_next_init = w.clone()
        w_avg, w_inner_history, inner_f_traj = projected_subgradient_inner_loop(
            S, w_next_init, P_list, pi_list, d, states, state_to_idx, K, eta
        )
        
        # Update weight for next iteration
        w = w_avg
        history['w_after_psg'].append(w.clone())
        
        if verbose:
            f_current = calculate_f(S, w, P_list, pi_list, d, states, state_to_idx)
            print(f"  After inner loop: f(S, w_avg) = {f_current.item():.6f}")
        
        # Outer loop: Distorted greedy step
        S_new, best_j, best_e, best_gain = distorted_greedy_step(
            S, w, P_list, pi_list, d, states, state_to_idx, V, l, i, beta
        )
        
        if verbose:
            if best_gain > 0:
                print(f"  Selected: e={best_e} for partition j={best_j}, gain={best_gain:.6f}")
            else:
                print(f"  No element selected (best gain={best_gain:.6f} <= 0)")

        # Update S
        S = S_new
        history['S_after'].append([s.copy() for s in S])

        # Record history
        if return_history:
            f_val = calculate_f(S, w, P_list, pi_list, d, states, state_to_idx)
            g_val = calculate_g(S, w, P_list, pi_list, d, states, state_to_idx, V, beta)
            c_val = calculate_c(S, w, P_list, pi_list, d, states, state_to_idx, V, beta)
            
            history['f_values'].append(f_val.item())
            history['g_values'].append(g_val.item())
            history['c_values'].append(c_val.item())
            history['S_history'].append([s.copy() for s in S])
            history['w_history'].append(w_inner_history)
            history['w_avg_history'].append(w.clone())
            history['inner_f_values'].append(inner_f_traj)
            history['selected_elements'].append((best_j, best_e))
            history['gains'].append(best_gain)
        
        if verbose:
            print()
    
    # Final evaluation
    f_final = calculate_f(S, w, P_list, pi_list, d, states, state_to_idx)
    
    if verbose:
        print(f"Final Results")
        print(f"=============")
        print(f"Final S: {S}")
        print(f"Final |supp(S)|: {sum(len(s) for s in S)}")
        print(f"Final w: {w}")
        print(f"Final f(S, w): {f_final.item():.6f}")
    
    if return_history:
        return S, w, f_final, history
    else:
        return S, w, f_final
