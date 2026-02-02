from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Suppress warning about flex_attention without torch.compile (we intentionally use uncompiled for multi-GPU inference)
torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from flash_attn import flash_attn_func


def flex_octopus_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    log_gated_states: torch.Tensor,
    scaling: float,
    q_start_pos: int = 0,
) -> torch.Tensor:
    """
    Octopus attention with gating mechanism.
    
    Args:
        query_states: Query tensor [batch, num_heads, q_len, head_dim]
        key_states: Key tensor [batch, num_kv_heads, kv_len, head_dim]
        value_states: Value tensor [batch, num_kv_heads, kv_len, head_dim]
        log_gated_states: Log-sigmoid gate scores [batch, num_kv_heads, kv_len]
        scaling: Attention scaling factor
        q_start_pos: The absolute position of the first query token (for causal masking during generation)
    """
    q_len = query_states.shape[2]
    kv_len = key_states.shape[2]
    
    def octopus_score_mod(score, b, h, q_idx, kv_idx):
        num_heads = query_states.shape[1]
        num_kv_heads = key_states.shape[1]
        group_size = num_heads // num_kv_heads
        kv_head = h // group_size
        
        # log_beta is (Batch, KV_Heads, SeqLen)
        log_gated_state = log_gated_states[b, kv_head, kv_idx]
        
        return score + log_gated_state

    # Causal mask: query at absolute position (q_start_pos + q_idx) can attend to kv at position kv_idx
    # For prefill: q_start_pos=0, so this is standard causal (q_idx >= kv_idx)
    # For generation: q_start_pos=cache_len, so query can attend to all previous positions
    block_mask = create_block_mask(
        lambda b, h, q, k: (q_start_pos + q) >= k,
        B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len,
        device=query_states.device,
    )

    attention_output = flex_attention(
        query_states, key_states, value_states,
        score_mod=octopus_score_mod,
        block_mask=block_mask,
        scale=scaling,
        enable_gqa=True,
    )
    attention_output = attention_output.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
    return attention_output


compiled_flex_octopus_attention = torch.compile(flex_octopus_attention, dynamic=False)

# Use uncompiled version for multi-GPU inference to avoid recompilation overhead
uncompiled_flex_octopus_attention = flex_octopus_attention


def get_q_start_pos(cache_position: Optional[torch.Tensor]) -> int:
    """Get the starting position for queries based on cache_position."""
    if cache_position is None:
        return 0
    return int(cache_position[0].item())


def flash_attention_2(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    return flash_attn_func(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        softmax_scale=scaling,
        causal=True,
    )

