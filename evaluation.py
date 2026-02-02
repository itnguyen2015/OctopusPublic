import torch

import transformers
from transformers import AutoTokenizer

from octopus.models import OctopusQwen3ForCausalLM, OctopusLlamaForCausalLM

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate

from src.octopus.cache_utils_pruned import OctopusPrunedCache


def get_model_class(pretrained: str):
    """Return the appropriate model class based on pretrained path/name."""
    pretrained_lower = pretrained.lower()
    if "llama" in pretrained_lower:
        return OctopusLlamaForCausalLM
    elif "qwen" in pretrained_lower:
        return OctopusQwen3ForCausalLM
    else:
        raise ValueError(f"Unknown model type for: {pretrained}. Supported: llama, qwen")


@register_model("octopus")
class OctopusEvalWrapper(HFLM):

    def __init__(self, pretrained="checkpoints/llama-8b-alpaca-cleaned", max_length=2048, batch_size=None, device="cuda",
                 dtype=torch.float16):
        LM.__init__(self)
        model_class = get_model_class(pretrained)
        self._model = model_class.from_pretrained(pretrained, device=device, dtype=dtype)
        self.AUTO_MODEL_CLASS = model_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)
        self._model.config.use_base_attention = True
        self._model.config.kv_cache_budget = 128
        self._model.config.kv_cache_recent_window = 32
        self._model.config.kv_cache_sink_tokens = 4

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            return self.model.generate(
                input_ids=context,
                max_length=max_length,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                past_key_values=OctopusPrunedCache(),
            )


if __name__ == "__main__":
    cli_evaluate()