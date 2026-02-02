from transformers.models.llama.configuration_llama import LlamaConfig


class OctopusLlamaConfig(LlamaConfig):
    model_type = "octopus_llama"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_base_attention = kwargs.get("use_base_attention", False)
        # KV cache pruning settings
        # budget: total number of tokens to keep per KV head (None = no pruning)
        # recent_window: always keep the last N tokens (for recent context)
        # sink_tokens: always keep the first N tokens (for system prompt / attention sinks)
        self.kv_cache_budget = kwargs.get("kv_cache_budget", None)
        self.kv_cache_recent_window = kwargs.get("kv_cache_recent_window", 64)
        self.kv_cache_sink_tokens = kwargs.get("kv_cache_sink_tokens", 4)

