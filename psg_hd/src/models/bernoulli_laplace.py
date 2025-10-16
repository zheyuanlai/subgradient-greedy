import torch
from itertools import combinations
import itertools
import math

def falling(a, k):
    if k == 0:
        return 1
    v = 1
    for i in range(k):
        v *= (a - i)
    return v

def bernoulli_laplace_level_spectral(d, s=1, device='cpu'):
    """Exact spectral construction for Bernoulli-Laplace level model restricted to {0,1}^d.

    Parameters correspond to: N = d, l_1=...=l_d = 1, l_{d+1}= d (so total L = 2d).
    State x in {0,1}^d; implicit x^{d+1} = d - |x|. Stationary distribution:
        π(x) ∝ prod_{i=1}^d C(1, x_i) * C(d, d - |x|) = C(d, d-|x|).
    Eigenvalues (Theorem 4.19 specialized):
        β_n = Σ_{k=0}^n C(n,k) * ( (N - s)_{[n-k]} * s_{[k]} ) / ( N_{[n-k]} * (L - N)_{[k]} )
      where N = d, L - N = d, so denominator second term uses d.
    Eigenfunctions depend only on r = |x| (Hamming weight). We construct orthonormal basis
    {φ_n(r)} via Gram-Schmidt over polynomial basis 1, r, r^2, ... under weight w_r = C(d, r) C(d, d-r).

    P(x,y) = Σ_{n=0}^d β_n φ_n(|x|) φ_n(|y|) π(y).
    """
    N = d
    L_minus_N = d
    # Enumerate states
    states = list(itertools.product([0,1], repeat=d))
    n_states = len(states)
    state_to_idx = {s:i for i,s in enumerate(states)}
    # Precompute Hamming weights
    r_vals = torch.tensor([sum(s) for s in states], device=device)
    # Stationary distribution weights by r: w_r = C(d,r) * C(d, d-r)
    w_r = torch.tensor([math.comb(d, int(r)) * math.comb(d, d-int(r)) for r in range(d+1)], dtype=torch.float64, device=device)
    # Map each state to weight component
    pi_unnormalized = torch.tensor([w_r[int(r)].item() for r in r_vals], dtype=torch.float64, device=device)
    pi = (pi_unnormalized / pi_unnormalized.sum()).to(torch.float32)
    # Construct polynomial basis matrix B[r, m] = r^m
    R = torch.arange(0, d+1, device=device, dtype=torch.float64)
    max_deg = d
    poly = torch.stack([R**m for m in range(max_deg+1)], dim=1)  # (d+1, d+1)
    # Gram-Schmidt with weighting w_r to get orthonormal φ_m(r)
    ortho = []
    norms = []
    for m in range(max_deg+1):
        v = poly[:, m].clone()
        for j in range(len(ortho)):
            proj = (w_r * v * ortho[j]).sum() * ortho[j]
            v = v - proj
        norm = torch.sqrt((w_r * v * v).sum())
        if norm < 1e-14:
            # Degenerate; stop
            break
        v = v / norm
        ortho.append(v)
        norms.append(norm)
    phi = torch.stack(ortho, dim=1)  # shape (d+1, M) M<=d+1
    M = phi.shape[1]
    # Compute eigenvalues β_n for n=0..M-1 (formula defined up to N)
    beta = []
    for n in range(M):
        acc = 0.0
        for k in range(n+1):
            num = falling(N - s, n - k) * falling(s, k)
            den = falling(N, n - k) * falling(L_minus_N, k)
            acc += math.comb(n, k) * (num / den)
        beta.append(acc)
    beta = torch.tensor(beta, dtype=torch.float64, device=device)
    # Build P via spectral sum: P(x,y) = Σ β_n φ_n(r_x) φ_n(r_y) π(y)
    P = torch.zeros((n_states, n_states), dtype=torch.float64, device=device)
    # Precompute phi(|x|)
    phi_r = phi[r_vals.long()]  # (n_states, M)
    for n in range(M):
        outer = torch.outer(phi_r[:, n], phi_r[:, n])  # φ_n(r_x) φ_n(r_y)
        P += beta[n] * outer * pi[None, :].to(torch.float64)
    # Numerical fixes: enforce non-negativity & stochasticity
    P = torch.clamp(P, min=0)
    row_sums = P.sum(dim=1, keepdim=True)
    P = P / row_sums
    return P.to(torch.float32), states, state_to_idx, pi

def bernoulli_laplace_level_model(d, s=1, device='cpu'):
    """Approximate Bernoulli-Laplace level model restricted to {0,1}^d with N=d and
    l_1=...=l_d=1, l_{d+1}=d. We treat states as binary vectors x in {0,1}^d and
    implicitly x^{d+1}= d - sum_i x^i. We construct a reversible chain with target π(x)
    proportional to product_{i=1}^d C(1, x^i) * C(d, d - sum x^i) = C(d, d - |x|)
    simplifying the multivariate hypergeometric normalization.

    Transition: choose s coordinates with value 1 and s with value 0 (if possible) uniformly
    and swap them (turn ones to zeros and zeros to ones) — generalized multi-swap. If not enough
    ones or zeros, fall back to single-bit flip maintaining reversibility via Metropolis ratio using π.
    """
    states = list(itertools.product([0,1], repeat=d))
    n_states = len(states)
    state_to_idx = {s:i for i,s in enumerate(states)}
    # Compute unnormalized pi weights: w(x) = C(d, d-|x|)
    weights = []
    for st in states:
        ones = sum(st)
        w = math.comb(d, d - ones)
        weights.append(w)
    weights = torch.tensor(weights, dtype=torch.float64, device=device)
    pi = weights / weights.sum()
    P = torch.zeros((n_states, n_states), dtype=torch.float64, device=device)
    for i, st in enumerate(states):
        ones_idx = [k for k,b in enumerate(st) if b==1]
        zeros_idx = [k for k,b in enumerate(st) if b==0]
        proposals = []
        if len(ones_idx) >= s and len(zeros_idx) >= s:
            # all combinations of s ones and s zeros
            for ones_choose in combinations(ones_idx, s):
                for zeros_choose in combinations(zeros_idx, s):
                    lst = list(st)
                    for o in ones_choose: lst[o] = 0
                    for z in zeros_choose: lst[z] = 1
                    proposals.append(tuple(lst))
        else:
            # fallback single bit flip
            for k in range(d):
                lst = list(st)
                lst[k] = 1 - lst[k]
                proposals.append(tuple(lst))
        proposals = list(set([p for p in proposals if p!=st]))
        if not proposals:
            P[i,i]=1.0
            continue
        base_prob = 1.0 / len(proposals)
        for p in proposals:
            j = state_to_idx[p]
            # Metropolis-Hastings acceptance keeping uniform proposal symmetric
            acc = min(1.0, (pi[j]/pi[i]).item())
            P[i,j] = base_prob * acc
        stay = 1.0 - P[i].sum().item()
        P[i,i] = max(stay, 0.0)
    # Normalize rows for safety
    P = P / P.sum(dim=1, keepdim=True)
    return P.to(torch.float32), states, state_to_idx, pi.to(torch.float32)

def generate_states(d, k):
    """
    Generate all possible states for the Bernoulli-Laplace model.
    A state is represented by a tuple of 0s and 1s of length d,
    with exactly k ones.
    
    Args:
        d (int): Total number of particles
        k (int): Number of red particles (ones)
        
    Returns:
        list: List of all valid states
    """
    states = []
    for combo in combinations(range(d), k):
        state = [0] * d
        for i in combo:
            state[i] = 1
        states.append(tuple(state))
    return states

def bernoulli_laplace_model(d, k, device='cpu'):
    """
    Constructs the transition matrix for the Bernoulli-Laplace model.
    
    In this model, we have d particles, k of which are red (1) and d-k are black (0).
    At each step, we pick one particle from each color and swap them.

    Args:
        d (int): Total number of particles.
        k (int): Number of red particles.
        device (str): The device to place tensors on.

    Returns:
        tuple: (P, states, state_to_idx)
            - P: The transition matrix
            - states: List of all states
            - state_to_idx: Dictionary mapping states to indices
    """
    states = generate_states(d, k)
    n_states = len(states)
    state_to_idx = {state: i for i, state in enumerate(states)}
    
    P = torch.zeros((n_states, n_states), device=device)

    for i, state_from in enumerate(states):
        zeros_indices = [j for j, bit in enumerate(state_from) if bit == 0]
        ones_indices = [j for j, bit in enumerate(state_from) if bit == 1]
        
        # For each possible swap
        for zero_idx in zeros_indices:
            for one_idx in ones_indices:
                state_to = list(state_from)
                state_to[zero_idx] = 1
                state_to[one_idx] = 0
                state_to_tuple = tuple(state_to)
                
                if state_to_tuple in state_to_idx:
                    j = state_to_idx[state_to_tuple]
                    # Probability of this swap: 1/(d*d)
                    P[i, j] += 1.0 / (d * d)

        # Probability of staying (picking same particle twice or invalid swaps)
        P[i, i] = 1.0 - P[i, :].sum()

    # Renormalize rows to sum to 1 to handle any floating point inaccuracies
    P = P / P.sum(dim=1, keepdim=True)

    return P, states, state_to_idx

def get_stationary_distribution(P):
    """
    Calculates the stationary distribution of a transition matrix P.
    
    Args:
        P (torch.Tensor): Transition matrix
        
    Returns:
        torch.Tensor: Stationary distribution pi
    """
    eigenvalues, eigenvectors = torch.linalg.eig(P.T)
    # Find the eigenvector corresponding to the eigenvalue 1
    stationary_vector = None
    for i in range(len(eigenvalues)):
        if torch.isclose(eigenvalues[i].real, torch.tensor(1.0, device=P.device), atol=1e-6):
            stationary_vector = eigenvectors[:, i].real
            break
    
    if stationary_vector is None:
        raise ValueError("Stationary distribution not found.")

    # Normalize the stationary distribution
    pi = stationary_vector / stationary_vector.sum()
    return pi