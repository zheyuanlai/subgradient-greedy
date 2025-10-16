import torch
import itertools

def curie_weiss_glauber(d, T=10.0, h=1.0, device='cpu'):
    """Curie-Weiss Glauber dynamics on {-1,+1}^d with Hamiltonian
        H(x) = - sum_{i,j} 1/2^{|j-i|} x_i x_j - h sum_i x_i.

    Transition: choose coordinate i uniformly, propose flip, accept with
        a = exp( - (H(y)-H(x))_+ / T ) / d  (Metropolis-style as specified)
    Then set P(x,x) to keep rows stochastic.
    Returns (P, states, state_to_idx, H_vals, pi) where pi is Gibbs at temperature T.
    """
    # Precompute coefficient matrix J_{ij} = 1/2^{|j-i|}
    idx = torch.arange(d, device=device)
    J = 1.0 / (2.0 ** (idx.unsqueeze(0) - idx.unsqueeze(1)).abs())
    # Enumerate all states in {-1,+1}^d
    states = list(itertools.product([-1, 1], repeat=d))
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}
    # Hamiltonian values
    H_vals = torch.zeros(n_states, device=device)
    for i, s in enumerate(states):
        x = torch.tensor(s, dtype=torch.float32, device=device)
        H_vals[i] = - (x @ J @ x) - h * x.sum()
    # Gibbs distribution
    H_scaled = -H_vals / T
    max_shift = H_scaled.max()
    weights = torch.exp(H_scaled - max_shift)  # numerical stability
    Z = weights.sum()
    pi = weights / Z
    # Build transition matrix
    P = torch.zeros((n_states, n_states), device=device)
    for i, s in enumerate(states):
        for coord in range(d):
            s_list = list(s)
            s_list[coord] *= -1  # flip spin
            y = tuple(s_list)
            j = state_to_idx[y]
            dH = H_vals[j] - H_vals[i]
            # acceptance probability per spec: (1/d) * exp(- (dH)_+ / T)
            accept = torch.exp(- torch.clamp(dH, min=0)/T) / d
            P[i, j] = accept
        P[i, i] = 1.0 - P[i].sum()
    # Normalize rows (safety)
    P = P / P.sum(dim=1, keepdim=True)
    return P, states, state_to_idx, H_vals, pi

def curie_weiss_model(d, beta, B, device='cpu'):
    """
    Constructs the transition matrix for the Curie-Weiss model using Glauber dynamics.

    Args:
        d (int): Number of spins.
        beta (float): Inverse temperature.
        B (float): External magnetic field.
        device (str): The device to place tensors on.

    Returns:
        tuple: (P, states, state_to_idx)
            - P: The transition matrix (tensor of shape [2^d, 2^d])
            - states: List of all states
            - state_to_idx: Dictionary mapping states to indices
    """
    # Generate all possible states (2^d binary vectors)
    states = [tuple(map(int, bin(i)[2:].zfill(d))) for i in range(2**d)]
    n_states = len(states)
    state_to_idx = {state: i for i, state in enumerate(states)}
    
    P = torch.zeros((n_states, n_states), device=device)

    for i, state_from in enumerate(states):
        for j in range(d):
            # Flip the j-th spin
            state_to = list(state_from)
            state_to[j] = 1 - state_to[j]
            state_to = tuple(state_to)
            
            k = state_to_idx[state_to]

            # Calculate energy change (magnetization)
            m = sum(2*s - 1 for s in state_from)  # Total magnetization
            delta_E = -(2 * state_from[j] - 1) * ((2/d) * m + B)
            
            # Transition probability using Glauber dynamics
            prob = (1 / d) * (1 / (1 + torch.exp(torch.tensor(beta * delta_E, device=device))))
            P[i, k] = prob

        # Diagonal: probability of staying in the same state
        P[i, i] = 1.0 - P[i, :].sum()

    # Renormalize rows to sum to 1 (handle floating point errors)
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
    # Find the eigenvector corresponding to eigenvalue 1
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