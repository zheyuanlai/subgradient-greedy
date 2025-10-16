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

def compute_B(P_list, partition, d, states, state_to_idx, pi_common=None, sample_w=None):
    """Compute (over-approximate) theoretical bound B as in manuscript.

    B = n * ( |X| * sup_{i,x,y} P_i(x,y) * log( P_i(x,y) / (⊗_k P_bar^(S_k)(x,y)) ) )^2.

    We approximate sup by iterating all (i,x,y). P_bar built from sample_w (default uniform).
    If pi_common provided, tensor product marginalization uses stationary distribution of P_bar re-derived anyway.
    """
    device = P_list[0].device
    n = len(P_list)
    n_states = P_list[0].shape[0]
    if sample_w is None:
        sample_w = torch.ones(n, device=device) / n
    # Build P_bar
    P_bar = sum(sample_w[i] * P_list[i] for i in range(n))
    # Use true stationary distribution of P_bar (ensures correctness for marginalization)
    pi_bar = stationary_distribution(P_bar)
    # Build tensor product of marginals
    P_bar_prod = compute_tensor_product(P_bar, pi_bar, partition, d, states, state_to_idx)
    epsilon = 1e-12
    sup_val = torch.tensor(0.0, device=device)
    for i, P in enumerate(P_list):
        # Avoid log(0) by clipping
        ratio = (P + epsilon) / (P_bar_prod + epsilon)
        # Only consider entries where P>0 as per definition
        mask = P > 0
        if mask.any():
            vals = P[mask] * torch.log(ratio[mask])
            local_max = vals.max()
            if local_max > sup_val:
                sup_val = local_max
    B = n * (n_states * sup_val) ** 2
    # Return scalar float for logging & usage
    return B.item()

def compute_tensor_product(P_bar, pi_bar, partition, d, states, state_to_idx):
    """
    Computes the tensor product ⊗_j P_bar^(S_j) for a given partition.
    
    CRITICAL: This is NOT a Kronecker product! The result operates on the ORIGINAL state space.
    For each pair of states (x, y), the transition probability is:
        P_product(x, y) = ∏_{k=1}^m P^(S_k)(x_{S_k}, y_{S_k})
    where x_{S_k} is the projection of state x onto coordinates in S_k.
    
    Args:
        P_bar (torch.Tensor): Weighted average transition matrix
        pi_bar (torch.Tensor): Weighted average stationary distribution
        partition (list of lists): Fixed partition [S_1, ..., S_m]
        d (int): Total number of coordinates
        states (list): List of all states
        state_to_idx (dict): Mapping from states to indices
        
    Returns:
        torch.Tensor: Tensor product matrix on the ORIGINAL state space
    """
    if not partition or len(partition) == 0:
        return P_bar
    
    n_states = len(states)
    device = P_bar.device
    
    # Step 1: Compute marginalized matrices P^(S_j) for each partition element
    marginal_matrices = []
    marginal_groupings = []  # Maps: state_idx -> marginal_state_idx for each S_j
    
    for S_j in partition:
        if len(S_j) == 0:
            continue
            
        # Get the marginalized transition matrix
        P_S_j = keep_S_in(P_bar, pi_bar, S_j, d, states, state_to_idx)
        marginal_matrices.append(P_S_j)
        
        # Create mapping: for each original state, what is its marginal state index in S_j?
        marginal_states_dict = {}
        marginal_state_counter = 0
        state_to_marginal_idx = []
        
        for state in states:
            # Project state onto S_j
            marginal_state = tuple(state[coord] for coord in S_j)
            if marginal_state not in marginal_states_dict:
                marginal_states_dict[marginal_state] = marginal_state_counter
                marginal_state_counter += 1
            state_to_marginal_idx.append(marginal_states_dict[marginal_state])
        
        marginal_groupings.append(torch.tensor(state_to_marginal_idx, device=device))
    
    # Step 2: Construct the product matrix on the original state space
    P_product = torch.ones((n_states, n_states), device=device)
    
    for x_idx in range(n_states):
        for y_idx in range(n_states):
            # For each partition element S_j, multiply the marginal transition probabilities
            for k, P_S_k in enumerate(marginal_matrices):
                # Get the marginal state indices for x and y in partition element S_k
                x_marginal_idx = marginal_groupings[k][x_idx].item()
                y_marginal_idx = marginal_groupings[k][y_idx].item()
                
                # Multiply the transition probability
                P_product[x_idx, y_idx] *= P_S_k[x_marginal_idx, y_marginal_idx]
    
    return P_product

def compute_h(w, P_list, pi_list, partition, d, states, state_to_idx, recompute_pi_bar=True):
    """
    Computes h(w) = -Σᵢ wᵢ · D_KL(Pᵢ || ⊗_j P̄^(S_j))
    This is the objective function we want to MINIMIZE.
    
    Args:
        w (torch.Tensor): Weight vector in simplex S_n
        P_list (list): List of transition matrices {P_1, ..., P_n}
        pi_list (list): List of stationary distributions
        partition (list of lists): Fixed partition [S_1, ..., S_m]
        d (int): Total number of coordinates
        states (list): List of all states
        state_to_idx (dict): Mapping from states to indices
        
    Returns:
        torch.Tensor: h(w) value
    """
    n = len(P_list)
    
    # Compute weighted average: P_bar = Σᵢ wᵢ Pᵢ
    P_bar = sum(w[i] * P_list[i] for i in range(n))
    # Option: recompute stationary distribution of P_bar rather than convex combo
    if recompute_pi_bar:
        pi_bar = stationary_distribution(P_bar)
    else:
        pi_bar = sum(w[i] * pi_list[i] for i in range(n))
    
    # Compute tensor product ⊗_j P_bar^(S_j) on the ORIGINAL state space
    P_bar_product = compute_tensor_product(P_bar, pi_bar, partition, d, states, state_to_idx)
    
    # Verify dimensions (should match now with correct implementation)
    assert P_bar_product.shape == P_list[0].shape, \
        f"Tensor product shape {P_bar_product.shape} doesn't match original {P_list[0].shape}"
    
    # Compute h(w) = -Σᵢ wᵢ · D_KL(Pᵢ || ⊗_j P_bar^(S_j))
    h_val = -sum(w[i] * kl_divergence(P_list[i], P_bar_product, pi_list[i]) for i in range(n))
    
    return h_val

def compute_subgradient(w, P_list, pi_list, partition, d, states, state_to_idx, recompute_pi_bar=True):
    """
    Computes the subgradient of h at w.
    
    From Theorem (Subgradient of h):
    gᵢ = D_KL(P_n || ⊗_k P̄(w)^(S_k)) - D_KL(Pᵢ || ⊗_k P̄(w)^(S_k))
    
    Args:
        w (torch.Tensor): Current weight vector in simplex S_n
        P_list (list): List of transition matrices {P_1, ..., P_n}
        pi_list (list): List of stationary distributions
        partition (list of lists): Fixed partition [S_1, ..., S_m]
        d (int): Total number of coordinates
        states (list): List of all states
        state_to_idx (dict): Mapping from states to indices
        
    Returns:
        torch.Tensor: Subgradient vector g of length n
    """
    n = len(P_list)
    device = P_list[0].device
    
    # Compute P_bar = Σᵢ wᵢ Pᵢ
    P_bar = sum(w[i] * P_list[i] for i in range(n))
    if recompute_pi_bar:
        pi_bar = stationary_distribution(P_bar)
    else:
        pi_bar = sum(w[i] * pi_list[i] for i in range(n))
    
    # Compute tensor product ⊗_j P_bar^(S_j) on the ORIGINAL state space
    P_bar_product = compute_tensor_product(P_bar, pi_bar, partition, d, states, state_to_idx)
    
    # Verify dimensions
    assert P_bar_product.shape == P_list[0].shape, \
        f"Tensor product shape {P_bar_product.shape} doesn't match original {P_list[0].shape}"
    
    # Compute D_KL(Pᵢ || ⊗_k P̄^(S_k)) for all i
    kl_divs = torch.zeros(n, device=device)
    for i in range(n):
        kl_divs[i] = kl_divergence(P_list[i], P_bar_product, pi_list[i])
    
    # Subgradient: gᵢ = D_KL(P_n || ⊗_k P̄^(S_k)) - D_KL(Pᵢ || ⊗_k P̄^(S_k))
    g = kl_divs[n-1] - kl_divs
    
    return g

def project_onto_simplex(v):
    """
    Projects vector v onto the probability simplex using efficient GPU-friendly algorithm.
    
    Based on: Duchi et al., "Efficient Projections onto the ℓ1-Ball for Learning in High Dimensions"
    
    Args:
        v (torch.Tensor): Vector to project (length n)
        
    Returns:
        torch.Tensor: Projected vector in simplex S_n
    """
    n = v.shape[0]
    device = v.device
    
    # Sort v in descending order
    u, _ = torch.sort(v, descending=True)
    
    # Find rho (largest j such that u[j] + (1/j+1)(1 - sum(u[0:j+1])) > 0)
    cssv = torch.cumsum(u, dim=0)
    rho = torch.nonzero((u * torch.arange(1, n+1, device=device) > (cssv - 1)), as_tuple=False)
    
    if rho.numel() > 0:
        rho = rho[-1].item()
    else:
        rho = n - 1
    
    # Compute threshold
    theta = (cssv[rho] - 1) / (rho + 1)
    
    # Project
    w = torch.clamp(v - theta, min=0)
    
    return w

def projected_subgradient_descent(w_init, P_list, pi_list, partition, d, states, state_to_idx,
                                   n_iter=100, eta=None, B=None, use_theoretical_eta=True,
                                   recompute_pi_bar=True, verbose=True, return_meta=False):
    """
    Algorithm: Projected Subgradient Descent (from manuscript)
    
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
        verbose (bool): Print iteration info
        
    Returns:
        tuple: (w_optimal, w_avg, history)
            - w_optimal: Final weight vector w^(t)
            - w_avg: Averaged weight vector w̄^t = (1/t)Σᵢ w^(i)
            - history: List of dictionaries with iteration info
    """
    n = len(P_list)
    device = P_list[0].device
    
    # Initialize
    w = w_init.clone()
    w_sum = torch.zeros(n, device=device)
    history = []
    
    # If using theoretical schedule and B not provided, compute an estimate once (uniform weights)
    B_est = None
    if use_theoretical_eta and B is None:
        B_est = compute_B(P_list, partition, d, states, state_to_idx)
        B = B_est
        if verbose:
            print(f"Computed B ≈ {B:.6e}")
    if not use_theoretical_eta and eta is None:
        eta = 0.1  # fallback heuristic
    
    if verbose:
        print(f"Starting projected subgradient descent with {n_iter} iterations...")
        print(f"Initial weights: {w.cpu().numpy()}")
    
    for t in range(1, n_iter + 1):
        # Compute subgradient at current w
        g = compute_subgradient(w, P_list, pi_list, partition, d, states, state_to_idx,
                                 recompute_pi_bar=recompute_pi_bar)

        if use_theoretical_eta:
            eta_t = (n / (B * t)) ** 0.5
        else:
            eta_t = eta
        
        # Subgradient update: v = w - η * g
        v = w - eta_t * g
        
        # Project onto simplex: w^(t+1) = argmin_{w' ∈ S_n} ||w' - v||²
        w = project_onto_simplex(v)
        
        # Accumulate for averaging
        w_sum += w
        
        # Compute objective h(w)
        h_val = compute_h(w, P_list, pi_list, partition, d, states, state_to_idx,
                          recompute_pi_bar=recompute_pi_bar)
        
        # Store history
        history.append({
            'iteration': t,
            'h(w)': h_val.item(),
            'weights': w.clone().cpu().numpy(),
            'grad_norm': torch.norm(g).item(),
            'step_size': float(eta_t)
        })
        
        if verbose and (t % max(1, n_iter // 10) == 0 or t == 1):
            print(f"Iter {t}/{n_iter}: h(w) = {h_val.item():.6f}, "
                  f"||g|| = {torch.norm(g).item():.6f}, "
                  f"w = {w.cpu().numpy()}")
    
    # Compute averaged weights
    w_avg = w_sum / n_iter
    
    if verbose:
        print(f"\nOptimization complete!")
        print(f"Final weights w^(t): {w.cpu().numpy()}")
        print(f"Averaged weights w̄^t: {w_avg.cpu().numpy()}")
        print(f"Final h(w): {history[-1]['h(w)']:.6f}")
    
    if return_meta:
        return w, w_avg, history, {'B': B}
    return w, w_avg, history