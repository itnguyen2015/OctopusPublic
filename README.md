## Installation

```bash
pip install -e .
```

**Requirements:** Python ≥3.12, PyTorch with CUDA, flash-attn, transformers, wandb

## Training

```bash
# Single node, 8 GPUs
torchrun --nproc_per_node=8 -m octopus.train
```

Edit `src/octopus/train.py` to configure:
- `MODEL_NAME` — Base model (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`)
- `DATASET_NAME` — Training dataset
- `TWO_PHASE_TRAINING` — Enable distillation → fine-tuning phases
- `GATE_LOSS_WEIGHT` — Sparsity regularization strength

## Evaluation

Evaluate with [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned,dtype=float16 \
  --tasks gsm8k --num_fewshot 5 \
  --batch_size 16
```

## How It Works

Each attention head learns a gate \( g \in (0,1) \) per key-value position. The gated attention score becomes:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \log g\right) V
\]

Training minimizes:
1. **Phase 1:** L2 distillation loss between gated and standard attention + gate sparsity
2. **Phase 2:** Language modeling cross-entropy + gate sparsity

At inference, tokens with low gate values are pruned from the KV cache.
