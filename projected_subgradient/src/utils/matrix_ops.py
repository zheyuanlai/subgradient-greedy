import torch

def get_marginal_indices(S, d, states, state_to_idx):
    """
    Get indices for marginalization over subset S.
    Groups states by their projection onto coordinates in S.
    
    Args:
        S (list): Subset of coordinates to marginalize over
        d (int): Total number of coordinates
        states (list): List of all states
        state_to_idx (dict): Mapping from states to indices
        
    Returns:
        list: List of lists, where each sublist contains indices of states
              that have the same projection onto S
    """
    marginal_states = {}
    for i, state in enumerate(states):
        # Project state onto coordinates in S
        marginal_state = tuple(state[j] for j in S)
        if marginal_state not in marginal_states:
            marginal_states[marginal_state] = []
        marginal_states[marginal_state].append(i)
    
    return list(marginal_states.values())

def marginalize(P, pi, S, d, states, state_to_idx):
    """
    Computes the marginalized transition matrix P_S.
    
    Args:
        P (torch.Tensor): Original transition matrix
        pi (torch.Tensor): Stationary distribution
        S (list): Subset of coordinates
        d (int): Total number of coordinates
        states (list): List of all states
        state_to_idx (dict): Mapping from states to indices
        
    Returns:
        torch.Tensor: Marginalized transition matrix
    """
    if not S:  # if S is empty
        return torch.ones((1, 1), device=P.device)

    S_indices = get_marginal_indices(S, d, states, state_to_idx)
    n_marginal_states = len(S_indices)
    P_S = torch.zeros((n_marginal_states, n_marginal_states), device=P.device)
    pi_S = torch.zeros(n_marginal_states, device=P.device)

    # Pre-calculate pi_S for all marginal states
    for i, indices_i in enumerate(S_indices):
        pi_S[i] = pi[indices_i].sum()

    # Calculate the marginalized transition matrix P_S
    for i, indices_i in enumerate(S_indices):
        for j, indices_j in enumerate(S_indices):
            if pi_S[i] > 0:
                # Sum of transition probabilities from each state in group i to all states in group j
                prob_to_j = P[indices_i, :][:, indices_j].sum(dim=1)
                # Weighted sum by the stationary distribution of states in group i
                weighted_prob = (pi[indices_i] * prob_to_j).sum()
                P_S[i, j] = weighted_prob / pi_S[i]
            
    return P_S

def keep_S_in(P, pi, S, d, states, state_to_idx):
    """
    Computes the keep-S-in transition matrix P^(S).
    This is the marginalized transition matrix on coordinates S.
    
    Args:
        P (torch.Tensor): Original transition matrix
        pi (torch.Tensor): Stationary distribution
        S (list): Subset of coordinates
        d (int): Total number of coordinates
        states (list): List of all states
        state_to_idx (dict): Mapping from states to indices
        
    Returns:
        torch.Tensor: Keep-S-in transition matrix
    """
    if not S:
        return torch.ones((1, 1), device=P.device)
    
    return marginalize(P, pi, S, d, states, state_to_idx)

def leave_S_out(P, pi, S, d, states, state_to_idx):
    """
    Computes the leave-S-out transition matrix P^(-S).
    This is the marginalized transition matrix on coordinates not in S.
    
    Args:
        P (torch.Tensor): Original transition matrix
        pi (torch.Tensor): Stationary distribution
        S (list): Subset of coordinates
        d (int): Total number of coordinates
        states (list): List of all states
        state_to_idx (dict): Mapping from states to indices
        
    Returns:
        torch.Tensor: Leave-S-out transition matrix
    """
    not_S = [i for i in range(d) if i not in S]
    return keep_S_in(P, pi, not_S, d, states, state_to_idx)

def kl_divergence(P, Q, pi):
    """
    Computes the KL divergence D_KL^pi(P || Q).
    
    D_KL^pi(P || Q) = sum_i pi_i * sum_j P_ij * log(P_ij / Q_ij)
    
    Args:
        P (torch.Tensor): First transition matrix
        Q (torch.Tensor): Second transition matrix
        pi (torch.Tensor): Stationary distribution
        
    Returns:
        torch.Tensor: KL divergence value
    """
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-12
    P_safe = P + epsilon
    Q_safe = Q + epsilon
    
    # Element-wise KL divergence
    kl_matrix = P_safe * (torch.log(P_safe) - torch.log(Q_safe))
    
    # Weighted sum by stationary distribution pi
    return (pi[:, None] * kl_matrix).sum()