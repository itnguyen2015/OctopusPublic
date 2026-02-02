"""
Pruned KV Cache implementation for Octopus models.

Unlike the standard OctopusDynamicCache which masks pruned tokens with -inf gates,
this implementation actually removes pruned tokens from memory, saving both memory
and computation during attention.
"""

from typing import Any, Optional

import torch

from transformers.cache_utils import Cache, DynamicLayer


class OctopusPrunedCacheLayer(DynamicLayer):
    """
    A cache layer that stores keys, values, and gates, and supports
    actual token removal during pruning.
    
    Unlike OctopusDynamicLayer which masks pruned tokens, this implementation
    removes them entirely from the cache tensors.
    """
    
    def __init__(self):
        super().__init__()
        self.gates: Optional[torch.Tensor] = None
    
    def lazy_initialization(self, key_states: torch.Tensor):
        """Initialize the cache with empty tensors matching the key states."""
        super().lazy_initialization(key_states)
        self.gates = torch.tensor([], dtype=self.dtype, device=self.device)
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Update the cache with new key, value, and gate states.
        
        Returns the full cached keys, values, and gates (including new additions).
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        
        gates = cache_kwargs.get("gates") if cache_kwargs is not None else None
        if gates is not None:
            self.gates = torch.cat([self.gates, gates], dim=-1)
        
        return self.keys, self.values, self.gates
    
    def get_seq_length(self) -> int:
        """Returns the current sequence length in the cache."""
        if not self.is_initialized or self.keys is None:
            return 0
        return self.keys.shape[-2]
    
    def prune_by_gate_scores(
        self,
        budget: int,
        recent_window: int = 0,
        sink_tokens: int = 0,
    ) -> None:
        """
        Prune this layer's KV cache by REMOVING low-importance tokens.
        
        Unlike the masking approach, this actually removes tokens from the cache,
        reducing memory usage and computation in subsequent attention operations.
        
        Strategy:
        1. Always keep the first `sink_tokens` (attention sinks / system prompt)
        2. Always keep the last `recent_window` tokens (recent context)
        3. From remaining middle tokens, keep top-K by gate score
        
        Args:
            budget: Total number of tokens to keep per KV head.
            recent_window: Number of recent tokens to always keep.
            sink_tokens: Number of initial tokens to always keep.
        """
        if not self.is_initialized or self.gates is None or self.gates.numel() == 0:
            return
        
        batch_size, num_kv_heads, seq_len = self.gates.shape
        
        # If budget >= seq_len, no pruning needed
        if budget >= seq_len:
            return
        
        # We need to select which tokens to keep
        # The selection is done per batch and per head
        
        # Build the keep mask
        keep_mask = torch.zeros(batch_size, num_kv_heads, seq_len, dtype=torch.bool, device=self.gates.device)
        
        # Always keep sink tokens
        if sink_tokens > 0:
            keep_mask[..., :sink_tokens] = True
        
        # Always keep recent tokens
        if recent_window > 0:
            keep_mask[..., -recent_window:] = True
        
        # Calculate how many tokens to select by gate score from the middle
        guaranteed_tokens = min(sink_tokens, seq_len) + min(recent_window, max(0, seq_len - sink_tokens))
        tokens_to_select = max(0, budget - guaranteed_tokens)
        
        if tokens_to_select > 0:
            # Define the middle region
            middle_start = sink_tokens
            middle_end = seq_len - recent_window if recent_window > 0 else seq_len
            
            if middle_end > middle_start:
                # Get gates for middle region
                middle_gates = self.gates[..., middle_start:middle_end]
                middle_len = middle_gates.shape[-1]
                
                # Select top-k from middle region
                k = min(tokens_to_select, middle_len)
                if k > 0:
                    _, top_indices = torch.topk(middle_gates, k=k, dim=-1, sorted=False)
                    # Adjust indices to global positions
                    top_indices = top_indices + middle_start
                    # Mark these positions
                    keep_mask.scatter_(-1, top_indices, True)
        
        # Now actually remove the tokens
        # We need to handle this carefully since different heads might want different tokens
        # For simplicity, we'll keep the union of tokens across all heads within a batch
        # This is a conservative approach that keeps more tokens but is simpler
        
        # Alternative: keep per-head selection (more memory efficient but more complex)
        # For now, let's do per-head selection since that's the whole point
        
        # Since we're doing per-head selection, we need to gather the kept tokens
        # This is tricky because different heads keep different tokens
        
        # For efficiency, let's use a simpler approach:
        # Take the union of kept tokens across heads (per batch item)
        # This ensures consistent sequence length across heads
        keep_mask_union = keep_mask.any(dim=1, keepdim=True).expand_as(keep_mask)
        
        # Count how many tokens we're keeping per batch
        num_kept = keep_mask_union[0, 0].sum().item()
        
        if num_kept >= seq_len:
            return  # No tokens to prune
        
        # Create indices for gathering
        # Shape: [batch, num_kv_heads, num_kept]
        kept_indices = keep_mask_union[0, 0].nonzero(as_tuple=True)[0]  # Same for all batch/heads
        
        # Gather the kept tokens
        # keys/values: [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_kv_heads, num_kept, head_dim]
        # gates: [batch, num_kv_heads, seq_len] -> [batch, num_kv_heads, num_kept]
        
        kept_indices_kv = kept_indices.view(1, 1, -1, 1).expand(batch_size, num_kv_heads, -1, self.keys.shape[-1])
        kept_indices_g = kept_indices.view(1, 1, -1).expand(batch_size, num_kv_heads, -1)
        
        self.keys = torch.gather(self.keys, dim=2, index=kept_indices_kv)
        self.values = torch.gather(self.values, dim=2, index=kept_indices_kv)
        self.gates = torch.gather(self.gates, dim=2, index=kept_indices_g)
    
    def reset(self) -> None:
        """Reset the cache to empty state."""
        if self.is_initialized:
            batch_size, num_kv_heads, _, head_dim = self.keys.shape
            self.keys = torch.empty(batch_size, num_kv_heads, 0, head_dim, dtype=self.dtype, device=self.device)
            self.values = torch.empty(batch_size, num_kv_heads, 0, head_dim, dtype=self.dtype, device=self.device)
            self.gates = torch.empty(batch_size, num_kv_heads, 0, dtype=self.dtype, device=self.device)


class OctopusPrunedCache(Cache):
    """
    A pruned KV cache that actually removes low-importance tokens from memory.
    
    Unlike OctopusDynamicCache which masks pruned tokens with -inf gates,
    this implementation removes them entirely, saving memory and computation.
    
    Example:
        ```python
        >>> from octopus.cache_utils_pruned import OctopusPrunedCache
        >>> cache = OctopusPrunedCache()
        >>> # Use in model generation with pruning enabled
        >>> # Tokens will be physically removed from cache when pruning occurs
        ```
    
    Note:
        When tokens are removed, the cache sequence length decreases. This is
        reflected in get_seq_length() and affects subsequent attention computations.
        The causal mask in attention must handle the reduced sequence correctly.
    """
    
    def __init__(self):
        # Initialize with our custom layer class
        super().__init__(layer_class_to_replicate=OctopusPrunedCacheLayer)
        self._seen_tokens = 0  # Track total tokens seen (not current cache size)
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Updates the cache with new key, value, and gate states.
        
        Returns the full cached tensors for this layer.
        """
        while len(self.layers) <= layer_idx:
            self.layers.append(OctopusPrunedCacheLayer())
        
        # Track seen tokens (for position calculations)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the current cache sequence length for the given layer."""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()
    
    def get_max_cache_shape(self) -> Optional[int]:
        """Returns None as this cache has dynamic size."""
        return None
    
    def get_gates(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Returns the cached gates for the given layer."""
        if layer_idx >= len(self.layers):
            return None
        return self.layers[layer_idx].gates
    
    def prune_by_gate_scores(
        self,
        budget: int,
        layer_idx: Optional[int] = None,
        recent_window: int = 0,
        sink_tokens: int = 0,
    ) -> None:
        """
        Prune the KV cache by REMOVING low-importance tokens from memory.
        
        This physically removes tokens from the cache, reducing memory usage
        and computation in subsequent attention operations.
        
        Args:
            budget: Total number of tokens to keep per KV head.
            layer_idx: If provided, only prune the specified layer.
            recent_window: Number of recent tokens to always keep.
            sink_tokens: Number of initial tokens to always keep.
        """
        if layer_idx is not None:
            if layer_idx < len(self.layers):
                self.layers[layer_idx].prune_by_gate_scores(budget, recent_window, sink_tokens)
        else:
            for layer in self.layers:
                layer.prune_by_gate_scores(budget, recent_window, sink_tokens)
    
    def __iter__(self):
        """Yields (keys, values, gates) for each layer."""
        for layer in self.layers:
            yield layer.keys, layer.values, layer.gates
    
    def __len__(self) -> int:
        """Returns the number of layers in the cache."""
        return len(self.layers)
    
    def reset(self) -> None:
        """Reset all layers to empty state."""
        for layer in self.layers:
            layer.reset()
        self._seen_tokens = 0

