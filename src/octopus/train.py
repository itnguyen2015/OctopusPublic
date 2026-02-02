import os
import torch
import torch.nn.functional as F
import torch.distributed as torch_dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
import wandb

from transformers import AutoTokenizer

from octopus.models import OctopusQwen3ForCausalLM, OctopusLlamaForCausalLM
from octopus.models.qwen3.modeling_octopus_qwen3 import OctopusQwen3DecoderLayer
from octopus.models.llama.modeling_octopus_llama import OctopusLlamaDecoderLayer

import octopus.distributed as dist
from octopus.logger import get_logger
from octopus.data import build_dataset
from octopus.optimizer import build_optimizer, build_scheduler
import octopus.utils as utils

def get_model_class(model_name: str):
    """Return the appropriate model class and decoder layer class based on model name."""
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower:
        return OctopusLlamaForCausalLM, OctopusLlamaDecoderLayer
    elif "qwen" in model_name_lower:
        return OctopusQwen3ForCausalLM, OctopusQwen3DecoderLayer
    else:
        raise ValueError(f"Unknown model type for: {model_name}. Supported: llama, qwen")


def setup_model(model_name, logger, gradient_checkpointing: bool = False):
    model_class, decoder_layer_class = get_model_class(model_name)
    logger.info(f"Using model class: {model_class.__name__}, decoder layer: {decoder_layer_class.__name__}")
    
    model = model_class.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="eager",
    )
    logger.info(f"Model config: {model.config}")
    
    for name, param in model.named_parameters():
        if "gated_layer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    dist.setup_distributed()
    local_rank = dist.get_local_rank()
    torch.cuda.set_device(local_rank)
    
    dist.apply_fdsp_2(model, modules_to_shard=[decoder_layer_class], activation_checkpointing=gradient_checkpointing)
    
    # # Compile AFTER FSDP
    # model = torch.compile(
    #     model,
    #     mode="max-autotune",      # or "reduce-overhead"
    #     fullgraph=False,
    #     dynamic=False,
    # )

    logger.info(
        f"Model size: {utils.get_model_size(model)} | "
        f"Number of parameters: {utils.get_num_params(model) / 1e9:.2f}B | "
        f"Trainable parameters: {utils.get_trainable_params(model) / 1e6:.2f}M"
    )
    
    return model


def save_consolidated_checkpoint(tokenizer, model, output_dir: str, logger, cpu_offload: bool = True):
    logger.info(f"Saving consolidated checkpoint to {output_dir}")
    
    # DCP requires options to gather the full state dict to CPU
    sd_options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload)
    state_dict, _ = get_state_dict(model, optimizers=(), options=sd_options)
    
    if dist.get_global_rank() == 0:
        model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)


def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    logger,
    num_epochs,
    attn_loss_weight: float,
    gate_loss_weight: float,
    grad_clip: float,
    total_steps: int,
    use_distillation: bool = True,
    two_phase_training: bool = False,
    phase1_epochs: int = 1,
    log_interval: int = 10,
):
    def compute_l2_loss(attention_outputs):
        """Average L2 loss between octopus and softmax attentions for layers with both."""
        l2_loss = 0.0
        for octopus_attn, softmax_attn, _ in attention_outputs:
            l2_loss += F.mse_loss(octopus_attn, softmax_attn, reduction="mean")
        return l2_loss
    
    def regularize_gate_loss(attention_outputs):
        """Gate sparsity/entropy-style penalty: mean(g + g*(1-g)) over all gates."""
        gate_sum = 0.0
        numel = 0
        for _, _, log_gates in attention_outputs:
            gates = log_gates.exp()  # logsigmoid -> sigmoid
            gate_sum += (gates + gates * (1 - gates)).sum()
            numel += gates.numel()
        return gate_sum / max(numel, 1)
    
    def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
        if not torch_dist.is_available() or not torch_dist.is_initialized():
            return tensor
        tensor = tensor.clone()
        torch_dist.all_reduce(tensor, op=torch_dist.ReduceOp.AVG)
        return tensor
    
    device = torch.device("cuda", dist.get_local_rank()) if torch.cuda.is_available() else torch.device("cpu")
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        
        # Determine if we should use distillation for this epoch
        # Two-phase training: Phase 1 uses distillation, Phase 2 uses cross entropy only
        if two_phase_training:
            current_phase = 1 if epoch < phase1_epochs else 2
            use_distillation_this_epoch = (current_phase == 1)
            if dist.get_global_rank() == 0 and (epoch == 0 or epoch == phase1_epochs):
                phase_name = "Phase 1 (distillation + gate loss)" if current_phase == 1 else "Phase 2 (cross entropy + gate loss)"
                logger.info(f"Starting {phase_name} at epoch {epoch}")
        else:
            current_phase = None
            use_distillation_this_epoch = use_distillation
        
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_distillation_this_epoch:
                # Phase 1: use_base_attention = True
                # Loss = distillation + gate_loss (one forward pass)
                model.config.use_base_attention = True
                outputs = model(
                    batch["input_ids"],
                    output_attentions=True,
                    return_dict=True,
                )
                attn_l2_loss = compute_l2_loss(outputs.attentions)
                gate_sparse_loss = regularize_gate_loss(outputs.attentions)
                lm_loss = torch.tensor(0.0, device=device)
                
                total_loss = attn_loss_weight * attn_l2_loss + gate_loss_weight * gate_sparse_loss
            else:
                # Phase 2: use_base_attention = False
                # Loss = cross entropy + gate_loss (one forward pass)
                model.config.use_base_attention = False
                outputs = model(
                    batch["input_ids"],
                    labels=batch["input_ids"].clone(),
                    output_attentions=True,
                    return_dict=True,
                )
                attn_l2_loss = torch.tensor(0.0, device=device)
                gate_sparse_loss = regularize_gate_loss(outputs.attentions)
                lm_loss = outputs.loss
                
                total_loss = lm_loss + gate_loss_weight * gate_sparse_loss
            
            total_loss.backward()
            
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            
            # Metrics (reduce across ranks for consistent logging)
            lm_loss_val = reduce_mean(lm_loss.detach())
            attn_loss_val = reduce_mean(attn_l2_loss.detach())
            gate_loss_val = reduce_mean(gate_sparse_loss.detach())
            total_loss_val = reduce_mean(total_loss.detach())
            lr = scheduler.get_last_lr()[0]
            progress = 100.0 * (global_step + 1) / max(total_steps, 1)
            
            # Optional: monitor gating activation statistics
            gate_means = torch.stack([g.exp().mean() for _, _, g in outputs.attentions]).mean().detach()
            gate_means = reduce_mean(gate_means)
            
            if dist.get_global_rank() == 0:
                if global_step % log_interval == 0:
                    phase_str = f" | phase={current_phase}" if two_phase_training else ""
                    logger.info(
                        f"epoch={epoch} | step={step}{phase_str} | total_loss={total_loss_val.item():.4f} | "
                        f"lm_loss={lm_loss_val.item():.4f} | attn_l2_loss={attn_loss_val.item():.4f} | "
                        f"gate_loss={gate_loss_val.item():.4f} | lr={lr:.6f} | progress={progress:.2f}%"
                    )
                log_dict = {
                    "loss/total": total_loss_val.item(),
                    "loss/lm": lm_loss_val.item(),
                    "loss/attn_l2": attn_loss_val.item(),
                    "loss/gate_sparsity": gate_loss_val.item(),
                    "lr": lr,
                    "epoch": epoch,
                    "gates/mean": gate_means.item(),
                    "progress/percent": progress,
                    "step": global_step,
                }
                if two_phase_training:
                    log_dict["training/phase"] = current_phase
                wandb.log(log_dict, step=global_step)
            
            global_step += 1

def main():
    # MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    # DATASET_NAME = "open-r1/Mixture-of-Thoughts"
    DATASET_NAME = "unsloth/alpaca-cleaned"
    MAX_SEQ_LENGTH = 2048
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 0.01
    BETAS = (0.9, 0.95)
    EPS = 1e-8
    NUM_EPOCHS = 2
    WARMUP_RATIO = 0.05
    ATTN_LOSS_WEIGHT = 1
    GATE_LOSS_WEIGHT = 0.5
    GRAD_CLIP = 1.0
    USE_DISTILLATION = False
    TWO_PHASE_TRAINING = True  # Enable two-phase training
    PHASE1_EPOCHS = 1  # Number of epochs for Phase 1 (distillation + gate loss)
    GRADIENT_CHECKPOINTING = False
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "checkpoints/llama-8b-alpaca-cleaned")
    
    utils.set_seed()
    logger = get_logger()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = setup_model(MODEL_NAME, logger, GRADIENT_CHECKPOINTING)
    dataloader = build_dataset(DATASET_NAME, tokenizer, MAX_SEQ_LENGTH, BATCH_SIZE)
    optimizer = build_optimizer(model, LEARNING_RATE, WEIGHT_DECAY, BETAS, EPS)
    
    num_training_steps = len(dataloader) * NUM_EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    logger.info(f"Number of training steps: {num_training_steps} | Number of warmup steps: {num_warmup_steps}")
    scheduler = build_scheduler(optimizer, num_warmup_steps, num_training_steps, MIN_LR)
    
    if dist.get_global_rank() == 0:
        config = {
                "model": MODEL_NAME,
                "dataset": DATASET_NAME,
                "max_seq_length": MAX_SEQ_LENGTH,
                "batch_size": BATCH_SIZE,
                "lr": LEARNING_RATE,
                "min_lr": MIN_LR,
                "weight_decay": WEIGHT_DECAY,
                "betas": BETAS,
                "eps": EPS,
                "num_epochs": NUM_EPOCHS,
                "warmup_ratio": WARMUP_RATIO,
                "attn_loss_weight": ATTN_LOSS_WEIGHT,
                "gate_loss_weight": GATE_LOSS_WEIGHT,
                "grad_clip": GRAD_CLIP,
                "use_distillation": USE_DISTILLATION,
                "two_phase_training": TWO_PHASE_TRAINING,
                "phase1_epochs": PHASE1_EPOCHS,
                "output_dir": OUTPUT_DIR,
            }
        logger.info(f"Wandb config: {config}")
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "octopus"),
            config=config,
        )
    
    train(
        model,
        dataloader,
        optimizer,
        scheduler,
        logger,
        NUM_EPOCHS,
        ATTN_LOSS_WEIGHT,
        GATE_LOSS_WEIGHT,
        GRAD_CLIP,
        num_training_steps,
        use_distillation=USE_DISTILLATION,
        two_phase_training=TWO_PHASE_TRAINING,
        phase1_epochs=PHASE1_EPOCHS,
    )
    
    if dist.get_global_rank() == 0:
        wandb.finish()
    
    save_consolidated_checkpoint(tokenizer, model, OUTPUT_DIR, logger, cpu_offload=False)
    dist.cleanup_distributed()
    
if __name__ == "__main__":
    main()