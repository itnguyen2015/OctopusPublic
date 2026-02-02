import os
from functools import lru_cache

import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
from torch.distributed._composable.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
)


@lru_cache()
def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))

@lru_cache()
def get_global_rank() -> int:
    return int(os.environ.get("RANK", 0))

@lru_cache()
def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))

def setup_distributed():
    local_rank = get_local_rank()
    global_rank = get_global_rank()
    world_size = get_world_size()

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl", 
        world_size=world_size, 
        rank=global_rank,
    )

    dist.barrier()

def cleanup_distributed():
    dist.destroy_process_group()
    
def apply_fdsp_2(
    model,
    modules_to_shard,
    activation_checkpointing: bool = False,
):
    if activation_checkpointing:
        apply_activation_checkpointing(model)
        
    world_size = get_world_size()
    mesh = DeviceMesh(device_type="cuda", mesh=[i for i in range(world_size)])
    
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )
    fsdp_config = {"mp_policy": mp_policy, "mesh": mesh, "reshard_after_forward": True}
    
    for module in model.modules():
        if any([isinstance(module, m) for m in modules_to_shard]):
            fully_shard(module, **fsdp_config)
    fully_shard(model, **fsdp_config)