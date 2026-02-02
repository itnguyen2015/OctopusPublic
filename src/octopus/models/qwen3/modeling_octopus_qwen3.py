from typing import Optional, Union

import torch
from torch import nn

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3MLP,
    Qwen3RotaryEmbedding,
    Qwen3Attention,
    GenerationMixin,
    Cache,
    GradientCheckpointingLayer,
    Qwen3PreTrainedModel,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, create_causal_mask, create_sliding_window_causal_mask

from octopus.cache_utils import OctopusDynamicCache
from .configuration_octopus_qwen3 import OctopusQwen3Config
from .attention import compiled_flex_octopus_attention, uncompiled_flex_octopus_attention, flash_attention_2, get_q_start_pos

class OctopusQwen3Attention(Qwen3Attention):
    """OctopusQwen3Attention is a wrapper around Qwen3Attention that adds octopus-specific functionality.
    """
    
    def __init__(self, config: OctopusQwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        
        self.gated_layer = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False, dtype=self.q_proj.weight.dtype, device=self.q_proj.weight.device),
            nn.SiLU(),
            nn.Linear(self.config.hidden_size, self.config.num_key_value_heads, bias=False, dtype=self.q_proj.weight.dtype, device=self.q_proj.weight.device),
        )
        for m in self.gated_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)
        
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        gated_states = self.gated_layer(hidden_states).transpose(1, 2) # (batch_size, num_kv_heads, seq_len)
        log_gated_states = nn.functional.logsigmoid(gated_states.float()).to(key_states.dtype)
        
        # 1. Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_values is not None:
            # Update cache with keys, values, and gates
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "gates": log_gated_states}
            key_states, value_states, log_gated_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        
        # Use uncompiled version during inference to avoid recompilation overhead with multi-GPU
        flex_fn = compiled_flex_octopus_attention if self.training else uncompiled_flex_octopus_attention
        # Get the starting position for queries (for correct causal masking during generation)
        q_start_pos = get_q_start_pos(cache_position)
        octopus_attention = flex_fn(query_states, key_states, value_states, log_gated_states, self.scaling, q_start_pos)
        
        softmax_attention = None
        if self.config.use_base_attention:
            softmax_attention = flash_attention_2(query_states, key_states, value_states, self.scaling) # (batch_size, seq_len, num_heads, head_dim)
        
        if self.config.use_base_attention:
            attention_output = softmax_attention.reshape(batch_size, seq_len, -1).contiguous()
        else:
            attention_output = octopus_attention.reshape(batch_size, seq_len, -1).contiguous()
            
        attention_output = self.o_proj(attention_output)
        
        return attention_output, (octopus_attention, softmax_attention, log_gated_states) 
        

class OctopusQwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OctopusQwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = OctopusQwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return (hidden_states, attention_outputs)

class OctopusQwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: OctopusQwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [OctopusQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = OctopusDynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        attentions = ()
        kv_cache_budget = getattr(self.config, "kv_cache_budget", None)
        kv_cache_recent_window = getattr(self.config, "kv_cache_recent_window", 64)
        kv_cache_sink_tokens = getattr(self.config, "kv_cache_sink_tokens", 4)
        
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, attention_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            attentions = attentions + (attention_outputs,)
            
            # Apply KV cache pruning based on gate scores if budget is specified
            if kv_cache_budget is not None and past_key_values is not None:
                past_key_values.prune_by_gate_scores(
                    budget=kv_cache_budget,
                    layer_idx=layer_idx,
                    recent_window=kv_cache_recent_window,
                    sink_tokens=kv_cache_sink_tokens,
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            attentions=attentions,
        )
        
class OctopusQwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _no_split_modules = ["OctopusQwen3DecoderLayer"]
    
    def __init__(self, config):
        super().__init__(config)
        self.model = OctopusQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.config.use_base_attention = True
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )