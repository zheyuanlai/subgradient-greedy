import torch

def get_device():
    """
    Returns the appropriate device (GPU if available, otherwise CPU).
    
    Returns:
        torch.device: The device to use for computations
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_gpu(tensor):
    """
    Transfer a tensor to the GPU if available.
    
    Args:
        tensor: The tensor to transfer
        
    Returns:
        The tensor on GPU if available, otherwise on CPU
    """
    if torch.cuda.is_available():
        return tensor.to('cuda')
    return tensor

def from_gpu(tensor):
    """
    Transfer a tensor from the GPU to the CPU.
    
    Args:
        tensor: The tensor to transfer
        
    Returns:
        The tensor on CPU
    """
    if tensor.is_cuda:
        return tensor.cpu()
    return tensor

def gpu_available():
    """
    Check if GPU is available.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    return torch.cuda.is_available()