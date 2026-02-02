import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float,
):
    
    return get_cosine_with_min_lr_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr=min_lr,
    )
    
def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    is_fused: bool = False,
):
    # Parse betas if it's a string
    if isinstance(betas, str):
        betas = eval(betas)
    
    return AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        fused=is_fused, # TODO: remove this
    )