from .qwen3.modeling_octopus_qwen3 import OctopusQwen3ForCausalLM
from .qwen3.configuration_octopus_qwen3 import OctopusQwen3Config
from .llama.modeling_octopus_llama import OctopusLlamaForCausalLM
from .llama.configuration_octopus_llama import OctopusLlamaConfig

__all__ = [
    "OctopusQwen3ForCausalLM",
    "OctopusQwen3Config",
    "OctopusLlamaForCausalLM",
    "OctopusLlamaConfig",
]