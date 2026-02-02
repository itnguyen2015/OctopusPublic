from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Suppress warning about flex_attention without torch.compile (we intentionally use uncompiled for multi-GPU inference)
torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
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
    attention_output = attention_output.transpose(1, 2).contiguous() # (batch_size, seq_len, num_heads, head_dim)
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

# class OctopusAttention(nn.Module):
#     """
#     OctopusAttention is a wrapper around a base attention module that adds octopus-specific functionality.
#     """
#     def __init__(self, base_attention: nn.Module, init_gated_layer: bool = False, use_base_attention: bool = False):
#         super().__init__()
#         self.base_attention = base_attention
#         self.use_base_attention = use_base_attention
#         self._initialize_from_base_attention()
        
#         self.gated_layer = nn.Sequential(
#             nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False, dtype=self.q_proj.weight.dtype, device=self.q_proj.weight.device),
#             nn.SiLU(),
#             nn.Linear(self.config.hidden_size, self.config.num_key_value_heads, bias=False, dtype=self.q_proj.weight.dtype, device=self.q_proj.weight.device),
#         )
        
#         if init_gated_layer:
#             for m in self.gated_layer.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.xavier_uniform_(m.weight)
                    
#         del self.base_attention
        
#     def _initialize_from_base_attention(self):
#         # Initialize qkvo projection
#         self.q_proj = self.base_attention.q_proj
#         self.k_proj = self.base_attention.k_proj
#         self.v_proj = self.base_attention.v_proj
#         self.o_proj = self.base_attention.o_proj
        
#         # Initialize qk norm
#         self.q_norm = self.base_attention.q_norm
#         self.k_norm = self.base_attention.k_norm
        
#         # Initialize other config
#         self.config = self.base_attention.config
#         self.layer_idx = self.base_attention.layer_idx
#         self.head_dim = self.base_attention.head_dim
#         self.scaling = self.base_attention.scaling
#         self.attention_dropout = self.base_attention.attention_dropout
#         self.sliding_window = self.base_attention.sliding_window
        
    
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         position_embeddings: tuple[torch.Tensor, torch.Tensor],
#         attention_mask: Optional[torch.Tensor],
#         past_key_values: Optional[Cache] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         **kwargs,
#     ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
#         batch_size, seq_len, hidden_size = hidden_states.shape
#         hidden_shape = (batch_size, seq_len, -1, self.head_dim)
        
#         query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
#         key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
#         value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
#         gated_states = self.gated_layer(hidden_states).transpose(1, 2) # (batch_size, num_kv_heads, seq_len)
#         log_gated_states = nn.functional.logsigmoid(gated_states.float()).to(key_states.dtype)
        
#         # 1. Apply RoPE
#         cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
#         if past_key_values is not None:
#             raise NotImplementedError("In training, past_key_values is not supported yet.")
#         octopus_attention = compiled_flex_octopus_attention(query_states, key_states, value_states, log_gated_states, self.scaling)
        
#         softmax_attention = None
#         if self.use_base_attention:
#             # FA2 uses non-transposed inputs
#             softmax_attention = flash_attn_func(
#                 query_states.transpose(1, 2),
#                 key_states.transpose(1, 2),
#                 value_states.transpose(1, 2),
#                 softmax_scale=self.scaling,
#                 causal=True,
#             ) # (batch_size, seq_len, num_heads, head_dim)
#             print("softmax_attention.shape:", softmax_attention.shape)
#             print("octopus_attention.shape:", octopus_attention.shape)
        
#         if self.use_base_attention:
#             attention_output = softmax_attention.reshape(batch_size, seq_len, -1).contiguous()
#         else:
#             attention_output = octopus_attention.reshape(batch_size, seq_len, -1).contiguous()
            
#         attention_output = self.o_proj(attention_output)
        
#         return attention_output, (octopus_attention, softmax_attention, log_gated_states)