"""
Projection utilities for optimization on constrained spaces.
"""
import torch


def project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    """
    Project a vector onto the probability simplex.
    
    Uses the algorithm from:
    Duchi et al. "Efficient Projections onto the l1-Ball for Learning in High Dimensions"
    
    Args:
        v: Input vector of shape (n,)
        
    Returns:
        Projected vector w such that w >= 0 and sum(w) = 1
    """
    n = v.shape[0]
    device = v.device
    
    # Sort v in descending order
    u, _ = torch.sort(v, descending=True)
    
    # Find the threshold
    cumsum = torch.cumsum(u, dim=0)
    rho_values = u - (cumsum - 1) / torch.arange(1, n + 1, device=device)
    rho = torch.where(rho_values > 0)[0][-1].item() + 1
    
    # Compute the threshold
    theta = (cumsum[rho - 1] - 1) / rho
    
    # Project
    w = torch.maximum(v - theta, torch.zeros_like(v))
    
    return w
