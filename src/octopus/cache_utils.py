from typing import Any, Optional

import torch

from transformers.cache_utils import Cache, DynamicLayer


class OctopusDynamicLayer(DynamicLayer):
    """
    A cache layer that grows dynamically like DynamicLayer, but also stores
    log sigmoid gates alongside keys and values.
    
    Gates have shape `[batch_size, num_kv_heads, seq_len]`.
    """
    
    def __init__(self):
        super().__init__()
        self.gates: Optional[torch.Tensor] = None
    
    def lazy_initialization(self, key_states: torch.Tensor):
        super().lazy_initialization(key_states)
        self.gates = torch.tensor([], dtype=self.dtype, device=self.device)
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Update the key, value, and gate caches in-place.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.
                Should contain 'gates' with shape [batch_size, num_kv_heads, seq_len].

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]: The key, value, and gate states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        
        # Handle gates if provided
        gates = cache_kwargs.get("gates") if cache_kwargs is not None else None
        if gates is not None:
            self.gates = torch.cat([self.gates, gates], dim=-1)
        return self.keys, self.values, self.gates
    
    def offload(self):
        """Offload this layer's data to CPU device."""
        super().offload()
        if self.is_initialized and self.gates is not None and self.gates.numel() > 0:
            self.gates = self.gates.to("cpu", non_blocking=True)
    
    def prefetch(self):
        """Move data back to the layer's device ahead of time."""
        super().prefetch()
        if self.is_initialized and self.gates is not None and self.gates.device != self.device:
            self.gates = self.gates.to(self.device, non_blocking=True)
    
    def reset(self) -> None:
        """Resets the cache values while preserving the objects."""
        super().reset()
        if self.is_initialized and self.gates is not None:
            self.gates.zero_()
    
    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders this layer's cache for beam search."""
        super().reorder_cache(beam_idx)
        if self.gates is not None and self.gates.numel() > 0:
            self.gates = self.gates.index_select(0, beam_idx.to(self.gates.device))
    
    def crop(self, max_length: int) -> None:
        """Crop the past key values up to a new `max_length`."""
        super().crop(max_length)
        if self.gates is not None and self.gates.numel() > 0:
            if max_length < 0:
                max_length = self.gates.shape[-1] - abs(max_length)
            self.gates = self.gates[..., :max_length]
    
    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension."""
        super().batch_repeat_interleave(repeats)
        if self.gates is not None and self.gates.numel() > 0:
            self.gates = self.gates.repeat_interleave(repeats, dim=0)
    
    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache."""
        super().batch_select_indices(indices)
        if self.gates is not None and self.gates.numel() > 0:
            self.gates = self.gates[indices, ...]

    def prune_by_gate_scores(self, budget: int, recent_window: int = 0, sink_tokens: int = 0) -> None:
        """
        Prune this layer's KV cache based on gate scores by masking out low-importance tokens.

        Strategy:
        1. Always keep the first `sink_tokens` (attention sinks / system prompt)
        2. Always keep the last `recent_window` tokens (recent context)
        3. From the remaining middle tokens, keep top-K by gate score where K = budget - sink_tokens - recent_window

        This ensures the model always has access to:
        - The beginning of the context (system prompt, user question)
        - Recent tokens (what was just generated)
        - Important tokens from the middle (selected by gate scores)

        Args:
            budget: Total number of tokens to keep per KV head.
            recent_window: Number of recent tokens to always keep (default 0).
            sink_tokens: Number of initial tokens to always keep (default 0).

        Note:
            - If budget >= seq_len, no pruning is performed.
            - If sink_tokens + recent_window >= budget, only those are kept (no gate-based selection).
        """
        if not self.is_initialized or self.gates is None or self.gates.numel() == 0:
            return

        # gates: [batch_size, num_kv_heads, seq_len]
        batch_size, num_kv_heads, seq_len = self.gates.shape

        # If budget >= seq_len, no pruning needed
        if budget >= seq_len:
            return

        # Create mask: True = keep, False = prune
        mask = torch.zeros_like(self.gates, dtype=torch.bool)

        # Always keep sink tokens (first N tokens)
        if sink_tokens > 0:
            mask[..., :sink_tokens] = True

        # Always keep recent tokens (last N tokens)
        if recent_window > 0:
            mask[..., -recent_window:] = True

        # Calculate how many tokens to select by gate score from the middle
        guaranteed_tokens = min(sink_tokens, seq_len) + min(recent_window, max(0, seq_len - sink_tokens))
        tokens_to_select = max(0, budget - guaranteed_tokens)

        if tokens_to_select > 0:
            # Define the middle region (excluding sink and recent)
            middle_start = sink_tokens
            middle_end = seq_len - recent_window if recent_window > 0 else seq_len

            if middle_end > middle_start:
                # Get gates for middle region only
                middle_gates = self.gates[..., middle_start:middle_end]
                middle_len = middle_gates.shape[-1]

                # Select top-k from middle region
                k = min(tokens_to_select, middle_len)
                if k > 0:
                    _, top_indices = torch.topk(middle_gates, k=k, dim=-1, sorted=False)
                    # Adjust indices to global positions
                    top_indices = top_indices + middle_start
                    # Mark these positions in the mask
                    mask.scatter_(-1, top_indices, True)

        # Set gate scores to -1e9 for pruned tokens
        neg_inf = torch.tensor(-1e9, dtype=self.gates.dtype, device=self.gates.device)
        self.gates = torch.where(mask, self.gates, neg_inf)


class OctopusDynamicCache(Cache):
    """
    A cache that grows dynamically and stores log sigmoid gates alongside keys and values.
    
    This is designed for Octopus attention where each token has a gating value that
    modifies attention scores.
    
    Example:

    ```python
    >>> from octopus.cache_utils import OctopusDynamicCache
    >>> past_key_values = OctopusDynamicCache()
    >>> # In attention forward:
    >>> cache_kwargs = {"gates": log_gated_states}  # shape: [batch, num_kv_heads, seq_len]
    >>> key_states, value_states, gates = past_key_values.update(
    ...     key_states, value_states, layer_idx, cache_kwargs
    ... )
    ```
    """
    
    def __init__(
        self,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        super().__init__(
            layer_class_to_replicate=OctopusDynamicLayer,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Updates the cache with the new `key_states`, `value_states`, and optionally `gates`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache. Should contain 'gates' for octopus attention.

        Return:
            A tuple containing the updated key, value, and gate states.
        """
        # Append layers as needed
        while len(self.layers) <= layer_idx:
            self.layers.append(OctopusDynamicLayer())
        
        if self.offloading:
            torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)
        
        keys, values, gates = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
        
        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)
        
        # if layer_idx == 0:
        #     exp_gates = torch.exp(gates)
        #     print("layer 0 exp gates:", exp_gates[0, -1, :])
        return keys, values, gates
    
    def get_gates(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Returns the cached gates for the given layer."""
        if layer_idx >= len(self.layers):
            return None
        return self.layers[layer_idx].gates
    
    def __iter__(self):
        """Yields (keys, values, gates) for each layer."""
        for layer in self.layers:
            yield layer.keys, layer.values, layer.gates

    def prune_by_gate_scores(
        self,
        budget: int,
        layer_idx: Optional[int] = None,
        recent_window: int = 0,
        sink_tokens: int = 0,
    ) -> None:
        """
        Prune the KV cache based on gate scores by masking out low-importance tokens.

        Strategy for each layer:
        1. Always keep the first `sink_tokens` (attention sinks / system prompt)
        2. Always keep the last `recent_window` tokens (recent context)
        3. From remaining middle tokens, keep top-K by gate score

        Args:
            budget: Total number of tokens to keep per KV head.
            layer_idx: If provided, only prune the specified layer. Otherwise,
                prune all layers in the cache.
            recent_window: Number of recent tokens to always keep.
            sink_tokens: Number of initial tokens to always keep.

        Example:
            ```python
            >>> cache = OctopusDynamicCache()
            >>> # Keep 2048 tokens: 4 sink + 64 recent + top 1980 by gate score
            >>> cache.prune_by_gate_scores(budget=2048, recent_window=64, sink_tokens=4)
            ```

        Note:
            - If budget >= seq_len for a layer, that layer is not pruned.
            - Gate scores indicate token importance (higher = more important).
        """
        if layer_idx is not None:
            if layer_idx < len(self.layers):
                self.layers[layer_idx].prune_by_gate_scores(budget, recent_window, sink_tokens)
        else:
            for layer in self.layers:
                layer.prune_by_gate_scores(budget, recent_window, sink_tokens)


def prune_kv_cache_by_gate_scores(
    cache: "OctopusDynamicCache",
    budget: int,
    layer_idx: Optional[int] = None,
) -> None:
    """
    Prune the KV cache based on gate scores by zeroing out low-importance tokens.

    This is a standalone utility function that wraps the cache's prune method.
    For each layer, it selects the top `budget` tokens per KV head with the highest
    gate scores and zeros out the key/value tensors for all other tokens.

    Args:
        cache: The OctopusDynamicCache instance to prune.
        budget: The number of tokens to keep per KV head. The effective cache
            capacity for each layer is `budget × num_kv_heads`.
        layer_idx: If provided, only prune the specified layer. Otherwise,
            prune all layers in the cache.

    Strategy:
        1. Compute token importance using the stored gate scores.
        2. Select the top `budget` tokens per KV head with highest gate scores.
        3. Keep the KV values for selected tokens unchanged.
        4. Set key and value tensors to zero for all other tokens.
        5. Preserve original tensor shapes and ordering.

    Example:
        ```python
        >>> from octopus.cache_utils import OctopusDynamicCache, prune_kv_cache_by_gate_scores
        >>> cache = OctopusDynamicCache()
        >>> # After populating cache during forward pass...
        >>> # Prune to keep top 2048 tokens per KV head
        >>> prune_kv_cache_by_gate_scores(cache, budget=2048)
        ```

    Note:
        - If budget >= seq_len for a layer, that layer is not pruned.
        - Gate scores indicate token importance (higher = more important).
        - Works for both prefilling and autoregressive generation phases.
        - Efficient for inference: uses torch.topk and scatter operations.
    """
    cache.prune_by_gate_scores(budget, layer_idx)
