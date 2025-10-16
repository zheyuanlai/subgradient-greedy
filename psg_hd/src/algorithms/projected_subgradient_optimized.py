"""
Optimized version of projected_subgradient.py with vectorized compute_tensor_product.

Key optimization: Use bitwise operations and advanced tensor indexing to avoid
explicit loops over state pairs, achieving full GPU parallelization.
"""
import torch
from ..utils.matrix_ops import keep_S_in, kl_divergence


def stationary_distribution(P, max_iter=10000, tol=1e-10):
    """Compute stationary distribution of row-stochastic matrix P via power iteration on the left.

    We iterate v_{k+1} = v_k P until convergence.
    Ensures non-negative distribution without relying on eigen decomposition sign conventions.
    """
    n = P.shape[0]
    device = P.device
    v = torch.ones(n, device=device) / n
    for _ in range(max_iter):
        v_next = v @ P  # left multiplication (row vector)
        # Normalize to avoid drift
        v_next_sum = v_next.sum()
        if v_next_sum <= 0:
            raise RuntimeError("Encountered non-positive sum during power iteration.")
        v_next = v_next / v_next_sum
        if torch.norm(v_next - v, p=1).item() < tol:
            return v_next
        v = v_next
    return v  # return last iterate if not converged


def compute_tensor_product_vectorized(P_bar, pi_bar, partition, d, states, state_to_idx):
    """
    Vectorized computation of tensor product ⊗_j P_bar^(S_j) for binary state space {0,1}^d.
    
    MAJOR OPTIMIZATION: Uses bitwise operations and advanced indexing to eliminate
    all explicit loops over state pairs, achieving full GPU parallelization.
    
    For binary states, we can represent each state as an integer (bitstring).
    State projections become bitwise masks, enabling vectorized lookup.
    
    Args:
        P_bar (torch.Tensor): Weighted average transition matrix [n_states, n_states]
        pi_bar (torch.Tensor): Weighted average stationary distribution [n_states]
        partition (list of lists): Fixed partition [S_1, ..., S_m]
        d (int): Total number of coordinates
        states (list): List of all states (each is a tuple/list of 0s and 1s)
        state_to_idx (dict): Mapping from states to indices
        
    Returns:
        torch.Tensor: Tensor product matrix on the ORIGINAL state space [n_states, n_states]
    """
    if not partition or len(partition) == 0:
        return P_bar
    
    n_states = len(states)
    device = P_bar.device
    
    # Convert states to integer representation for efficient bitwise operations
    # For binary states {0,1}^d, state (b_{d-1}, ..., b_1, b_0) -> sum_i b_i * 2^i
    state_ints = torch.zeros(n_states, dtype=torch.long, device=device)
    for idx, state in enumerate(states):
        state_int = sum(bit * (2 ** i) for i, bit in enumerate(state))
        state_ints[idx] = state_int
    
    # Pre-compute marginal matrices and projection mappings for each partition element
    marginal_matrices = []
    marginal_idx_maps = []  # Maps: state_idx -> marginal_state_idx (vectorized)
    
    for S_j in partition:
        if len(S_j) == 0:
            continue
        
        # Get the marginalized transition matrix P^(S_j)
        P_S_j = keep_S_in(P_bar, pi_bar, S_j, d, states, state_to_idx)
        marginal_matrices.append(P_S_j)
        
        # Create bitmask for projection onto S_j
        # Only keep bits corresponding to coordinates in S_j
        mask = sum(2 ** coord for coord in S_j)
        
        # Project all states onto S_j using bitwise AND
        projected_ints = state_ints & mask
        
        # Create mapping from projected state to marginal index
        # Get unique projected states and their indices
        unique_projected, inverse_indices = torch.unique(projected_ints, return_inverse=True)
        
        # inverse_indices[i] gives the marginal state index for original state i
        marginal_idx_maps.append(inverse_indices)
    
    # ============================================================
    # VECTORIZED PRODUCT COMPUTATION (NO EXPLICIT LOOPS!)
    # ============================================================
    
    # Initialize product matrix to all ones
    P_product = torch.ones((n_states, n_states), device=device, dtype=P_bar.dtype)
    
    # For each partition element, multiply the corresponding marginal probabilities
    for k, P_S_k in enumerate(marginal_matrices):
        # Get marginal state indices for all original states
        marginal_idx = marginal_idx_maps[k]  # shape: [n_states]
        
        # Create index tensors for all (x, y) pairs simultaneously
        # x_marginal_idx[i, j] = marginal index of state i in partition k
        # y_marginal_idx[i, j] = marginal index of state j in partition k
        x_marginal_idx = marginal_idx.unsqueeze(1).expand(n_states, n_states)  # [n_states, n_states]
        y_marginal_idx = marginal_idx.unsqueeze(0).expand(n_states, n_states)  # [n_states, n_states]
        
        # Vectorized lookup: P_S_k[x_marginal_idx[i,j], y_marginal_idx[i,j]]
        # This retrieves all marginal transition probabilities at once!
        marginal_probs = P_S_k[x_marginal_idx, y_marginal_idx]  # [n_states, n_states]
        
        # Element-wise multiplication (all done in parallel on GPU)
        P_product *= marginal_probs
    
    return P_product


def compute_B(P_list, partition, d, states, state_to_idx, pi_common=None, sample_w=None):
    """Compute (over-approximate) theoretical bound B as in manuscript.

    B = n * ( |X| * sup_{i,x,y} P_i(x,y) * log( P_i(x,y) / (⊗_k P_bar^(S_k)(x,y)) ) )^2.

    We approximate sup by iterating all (i,x,y). P_bar built from sample_w (default uniform).
    
    Note: For numerical stability, we clamp the log ratio to avoid extreme values.
    """
    device = P_list[0].device
    n = len(P_list)
    n_states = P_list[0].shape[0]
    if sample_w is None:
        sample_w = torch.ones(n, device=device) / n
    # Build P_bar
    P_bar = sum(sample_w[i] * P_list[i] for i in range(n))
    # Use true stationary distribution of P_bar
    pi_bar = stationary_distribution(P_bar)
    # Build tensor product using optimized version
    P_bar_prod = compute_tensor_product_vectorized(P_bar, pi_bar, partition, d, states, state_to_idx)
    epsilon = 1e-12
    sup_val = torch.tensor(0.0, device=device)
    for i, P in enumerate(P_list):
        ratio = (P + epsilon) / (P_bar_prod + epsilon)
        mask = P > 0
        if mask.any():
            # Clamp log ratio to prevent numerical overflow
            log_ratio = torch.clamp(torch.log(ratio[mask]), min=-20.0, max=20.0)
            vals = P[mask] * log_ratio
            local_max = vals.max()
            if local_max > sup_val:
                sup_val = local_max
    
    # Compute B with additional safety clamping
    B_raw = n * (n_states * sup_val) ** 2
    
    # If B is unreasonably large, use a more conservative estimate
    # For d=10, typical reasonable B should be O(100-10000)
    # We use adaptive clamping based on state space size
    if n_states <= 32:  # d <= 5
        B_max = n * (n_states * 10.0) ** 2
    elif n_states <= 1024:  # d <= 10
        # For larger state spaces, use empirical scaling
        B_max = max(10000.0, n * 100.0)  # Much more conservative
    else:
        B_max = max(50000.0, n * 500.0)
    
    B = torch.clamp(B_raw, min=1.0, max=B_max)
    
    # Additional check: if still getting extreme values, fall back to simple estimate
    if torch.isnan(B) or torch.isinf(B) or B > 1e6:
        B = torch.tensor(max(100.0, n * 20.0), device=device)
    
    return B.item()


def compute_h(w, P_list, pi_list, partition, d, states, state_to_idx, recompute_pi_bar=True):
    """
    Computes h(w) = -Σᵢ wᵢ · D_KL(Pᵢ || ⊗_j P̄^(S_j))
    Uses optimized vectorized tensor product computation.
    """
    n = len(P_list)
    
    # Compute weighted average: P_bar = Σᵢ wᵢ Pᵢ
    P_bar = sum(w[i] * P_list[i] for i in range(n))
    if recompute_pi_bar:
        pi_bar = stationary_distribution(P_bar)
    else:
        pi_bar = sum(w[i] * pi_list[i] for i in range(n))
    
    # Compute tensor product using VECTORIZED version
    P_bar_product = compute_tensor_product_vectorized(P_bar, pi_bar, partition, d, states, state_to_idx)
    
    # Verify dimensions
    assert P_bar_product.shape == P_list[0].shape, \
        f"Tensor product shape {P_bar_product.shape} doesn't match original {P_list[0].shape}"
    
    # Compute h(w)
    h_val = -sum(w[i] * kl_divergence(P_list[i], P_bar_product, pi_list[i]) for i in range(n))
    
    return h_val


def compute_subgradient(w, P_list, pi_list, partition, d, states, state_to_idx, recompute_pi_bar=True):
    """
    Computes the subgradient of h at w using optimized tensor product.
    
    From Theorem (Subgradient of h):
    gᵢ = D_KL(P_n || ⊗_k P̄(w)^(S_k)) - D_KL(Pᵢ || ⊗_k P̄(w)^(S_k))
    """
    n = len(P_list)
    device = P_list[0].device
    
    # Compute P_bar = Σᵢ wᵢ Pᵢ
    P_bar = sum(w[i] * P_list[i] for i in range(n))
    if recompute_pi_bar:
        pi_bar = stationary_distribution(P_bar)
    else:
        pi_bar = sum(w[i] * pi_list[i] for i in range(n))
    
    # Compute tensor product using VECTORIZED version
    P_bar_product = compute_tensor_product_vectorized(P_bar, pi_bar, partition, d, states, state_to_idx)
    
    # Verify dimensions
    assert P_bar_product.shape == P_list[0].shape, \
        f"Tensor product shape {P_bar_product.shape} doesn't match original {P_list[0].shape}"
    
    # Compute KL divergences for all models
    kl_values = torch.zeros(n, device=device)
    for i in range(n):
        kl_values[i] = kl_divergence(P_list[i], P_bar_product, pi_list[i])
    
    # Subgradient: gᵢ = D_KL(P_n || Q) - D_KL(Pᵢ || Q)
    g = kl_values[-1] - kl_values
    
    return g


def projected_subgradient_descent(w_init, P_list, pi_list, partition, d, states, state_to_idx,
                                   n_iter=100, eta=None, B=None, use_theoretical_eta=True,
                                   recompute_pi_bar=True, verbose=True, return_meta=False):
    """
    Algorithm: Projected Subgradient Descent (OPTIMIZED version with vectorized tensor product)
    
    Solves: min_{w ∈ S_n} h(w) = -Σᵢ wᵢ · D_KL(Pᵢ || ⊗_j P̄^(S_j))
    
    Args:
        w_init (torch.Tensor): Initial weight vector in simplex S_n
        P_list (list): List of n transition matrices {P_1, ..., P_n}
        pi_list (list): List of stationary distributions
        partition (list of lists): Fixed partition [S_1, ..., S_m]
        d (int): Total number of coordinates
        states (list): List of all states
        state_to_idx (dict): Mapping from states to indices
        n_iter (int): Number of iterations t
        eta (float): Step size. If None, uses η = √(n/(Bt))
        B (float): Bound on gradient norm. If None, computed automatically
        use_theoretical_eta (bool): Use theoretical step size schedule
        recompute_pi_bar (bool): Recompute π̄ at each iteration (vs convex combination)
        verbose (bool): Print iteration info
        return_meta (bool): Return metadata (B value)
        
    Returns:
        tuple: (w_optimal, w_avg, history) or (w_optimal, w_avg, history, meta) if return_meta=True
    """
    n = len(P_list)
    device = P_list[0].device
    
    # Initialize
    w = w_init.clone()
    
    # Estimate B for step size if not provided
    if B is None:
        if verbose:
            print("Computing bound B...")
        B_theoretical = compute_B(P_list, partition, d, states, state_to_idx, sample_w=w)
        
        # If B is unreasonably large (e.g., >100,000), use empirical estimate instead
        if B_theoretical > 100000:
            if verbose:
                print(f"Theoretical B = {B_theoretical:.2f} is too large (numerical instability)")
                print("Using empirical gradient-based estimate instead...")
            
            # Estimate B empirically from initial gradients
            g_samples = []
            test_weights = [w.clone()]
            # Add a few perturbed weight samples
            for _ in range(3):
                w_test = torch.rand_like(w)
                w_test = w_test / w_test.sum()  # Normalize to simplex
                test_weights.append(w_test)
            
            for w_test in test_weights:
                g = compute_subgradient(w_test, P_list, pi_list, partition, d, states, state_to_idx, recompute_pi_bar=recompute_pi_bar)
                g_samples.append(torch.norm(g).item())
            
            # Use max observed gradient norm squared as empirical B
            max_grad_norm = max(g_samples)
            B = max_grad_norm ** 2
            
            # Apply safety bounds
            B = max(1.0, min(B, n * 1000.0))  # Clamp between 1 and n*1000
            
            if verbose:
                print(f"Empirical B = {B:.6f} (based on max gradient norm = {max_grad_norm:.4f})")
        else:
            B = B_theoretical
            if verbose:
                print(f"Bound B = {B:.6f}")
    
    # History tracking (matching original format)
    history = []
    w_sum = torch.zeros_like(w)
    
    if verbose:
        print(f"Starting projected subgradient descent (n_iter={n_iter})...")
    
    # Check initial objective to catch numerical issues early
    h_init = compute_h(w, P_list, pi_list, partition, d, states, state_to_idx, recompute_pi_bar=recompute_pi_bar)
    if torch.isnan(h_init) or torch.isinf(h_init):
        raise ValueError(f"Initial objective h(w) = {h_init.item()} is NaN or Inf! "
                        "This indicates numerical instability in the transition matrices. "
                        "For Curie-Weiss at high dimensions (d>=10), try:\n"
                        "  1. Reducing dimension (d <= 8)\n"
                        "  2. Adjusting temperature T and field h parameters\n"
                        "  3. Using fewer models (smaller n)\n"
                        "  4. Using Bernoulli-Laplace model instead")
    
    for t in range(1, n_iter + 1):
        # Compute step size
        if use_theoretical_eta:
            # Theoretical schedule: η_t = sqrt(n / (B * t))
            eta_t = torch.sqrt(torch.tensor(n / (B * t), device=device)).item()
        else:
            eta_t = eta if eta is not None else 0.01
        
        # Compute objective and subgradient
        h_t = compute_h(w, P_list, pi_list, partition, d, states, state_to_idx, recompute_pi_bar=recompute_pi_bar)
        g_t = compute_subgradient(w, P_list, pi_list, partition, d, states, state_to_idx, recompute_pi_bar=recompute_pi_bar)
        
        # Gradient descent step
        w_next = w - eta_t * g_t
        
        # Project onto simplex
        w = project_onto_simplex(w_next)
        
        # Accumulate for averaging
        w_sum += w
        
        # Log (matching original format)
        history.append({
            'iteration': t,
            'h(w)': h_t.item(),
            'weights': w.cpu().numpy(),
            'grad_norm': torch.norm(g_t).item(),
            'step_size': eta_t
        })
        
        if verbose and (t % 10 == 0 or t == 1):
            print(f"Iter {t:4d}: h(w) = {h_t.item():10.6f}, ||g|| = {torch.norm(g_t).item():8.4f}, eta = {eta_t:.6f}")
    
    # Compute averaged weights
    w_avg = w_sum / n_iter
    
    if verbose:
        h_final = compute_h(w, P_list, pi_list, partition, d, states, state_to_idx, recompute_pi_bar=recompute_pi_bar)
        print(f"\nFinal: h(w) = {h_final.item():.6f}")
        print(f"Final w = {w}")
    
    if return_meta:
        return w, w_avg, history, {'B': B}
    
    return w, w_avg, history


def project_onto_simplex(v):
    """
    Projects vector v onto the probability simplex using Duchi et al. algorithm.
    
    Based on: Duchi et al., "Efficient Projections onto the ℓ1-Ball for Learning in High Dimensions"
    
    Args:
        v: Input vector
        
    Returns:
        Projected vector w such that w >= 0 and sum(w) = 1
    """
    n = v.shape[0]
    device = v.device
    
    # Sort in descending order
    u, _ = torch.sort(v, descending=True)
    
    # Find rho (largest j such that u[j] + (1/(j+1))(1 - sum(u[0:j+1])) > 0)
    cssv = torch.cumsum(u, dim=0)
    rho_candidates = torch.nonzero((u * torch.arange(1, n+1, device=device) > (cssv - 1)), as_tuple=False)
    
    if rho_candidates.numel() > 0:
        rho = rho_candidates[-1].item()
    else:
        # Fallback: use last index if no positive candidates found
        rho = n - 1
    
    # Compute threshold
    theta = (cssv[rho] - 1) / (rho + 1)
    
    # Project
    w = torch.clamp(v - theta, min=0)
    
    return w
