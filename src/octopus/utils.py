import torch

def human_readable_size(num_bytes: int) -> str:
    """Converts a number of bytes into a human-readable string (e.g., "1.21 GB")."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    if num_bytes == 0:
        return "0 B"
    
    # Calculate the exponent for the unit
    exponent = int(torch.log2(torch.tensor(num_bytes, dtype=torch.float64)) / 10)
    # Ensure exponent is within the range of units
    exponent = min(exponent, len(units) - 1)
    
    size = num_bytes / (1024 ** exponent)
    unit = units[exponent]
    
    return f"{size:.2f} {unit}"

def get_model_size(model: torch.nn.Module):
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return human_readable_size(model_size)

def get_num_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())

def get_trainable_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False
        
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False