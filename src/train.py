"""
train.py
========
Training loop for the WordLSTM language model.

Features:
  - Mixed precision training (AMP) for speed + memory savings
  - Gradient accumulation for effective large batches
  - Gradient clipping (essential for RNN stability)
  - Cosine annealing LR schedule (simple, predictable decay)
  - Early stopping when validation loss plateaus (saves compute)
  - Validation every N steps with perplexity tracking
  - Checkpointing: best model + latest + periodic
  - Text generation samples at intervals
  - TensorBoard logging (optional)
  - Resume from checkpointUsage:
    python src/train.py --config configs/base.yaml
    python src/train.py --epochs 30 --batch_size 64 --lr 3e-4
"""

import os
import time
import math
import json
import argparse
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Project imports
from data import (
    Vocabulary,
    TextDataset,
    create_dataloaders,
)
from model import WordLSTM, ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Hyperparameters and paths for training."""

    # Data
    data_path: str = "data/raw/tinyshakespeare.txt"
    vocab_path: str = "data/processed/vocab.pkl"
    min_freq: int = 2
    val_ratio: float = 0.05
    test_ratio: float = 0.05

    # Model
    embedding_dim: int = 256
    hidden_size: int = 256
    num_layers: int = 2
    dropout_emb: float = 0.1
    dropout_in: float = 0.3
    dropout_out: float = 0.3
    tie_weights: bool = True

    # Training
    epochs: int = 30
    batch_size: int = 64
    seq_length: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    accumulation_steps: int = 1  # >1 for gradient accumulation

    # Scheduler (CosineAnnealingLR)
    scheduler_t_max: int = epochs  # number of epochs for a full cosine cycle
    scheduler_eta_min: float = 1e-6

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4

    # Evaluation
    eval_every_n_steps: int = 500
    sample_every_n_steps: int = 2000
    sample_prompt: str = "to be or not to"
    sample_tokens: int = 30

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    keep_last_n_checkpoints: int = 3

    # Mixed precision
    use_amp: bool = True

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Logging utility
# ─────────────────────────────────────────────────────────────────────────────

class TrainingLogger:
    """
    Simple logger that writes to both console and JSON file.

    Why not just print()? Because you want to:
      - Parse metrics later (e.g., plot loss curves)
      - Compare across runs
      - Resume reading logs programmatically

    JSON Lines format (.jsonl) is ideal — one JSON object per line,
    append-only, easy to load with pandas or jq.
    """

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode; truncate by deleting first if you want clean start
        self.file = open(self.log_path, 'a')
        self.start_time = time.time()

    def log(self, event: str, **metrics):
        record = {
            'event': event,
            'timestamp': time.time() - self.start_time,
            **metrics,
        }
        line = json.dumps(record)
        # Console: human-readable
        print(f"[{event:20s}] " + " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        ))
        # File: machine-readable
        self.file.write(line + "\n")
        self.file.flush()  # ensure we see logs immediately

    def close(self):
        self.file.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Perplexity
# ─────────────────────────────────────────────────────────────────────────────

def perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Perplexity = exp(loss). It can be interpreted as the effective
    branching factor — how many tokens the model is "confused between"
    on average. Lower is better.

    Random model: PPL = vocab_size
    Good model on Shakespeare: PPL ≈ 30-60
    Good model on WikiText-103: PPL ≈ 30-50
    """
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')


# ─────────────────────────────────────────────────────────────────────────────
# 4. Checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    model: WordLSTM,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    config: TrainingConfig,
    model_config: ModelConfig,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    val_loss: float,
    metrics: Optional[Dict] = None,
):
    """
    Save full training state. Includes everything needed to resume
    training exactly where you left off.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'training_config':      config.to_dict(),
        'model_config':         asdict(model_config),
        'epoch':                epoch,
        'global_step':          global_step,
        'best_val_loss':        best_val_loss,
        'val_loss':             val_loss,
        'metrics':              metrics or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: WordLSTM,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, Any]:
    """
    Load a checkpoint. Restores model, optimizer, scheduler, and metadata.
    Returns the metadata dict (epoch, step, etc.) so the caller knows
    where training resumed from.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch':         checkpoint.get('epoch', 0),
        'global_step':   checkpoint.get('global_step', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
 }


def prune_old_checkpoints(checkpoint_dir: str, keep_n: int):
    """
    Delete old step checkpoints, keeping only the most recent N.

    We keep:
      - best.pt (always)
      - latest.pt (always)
      - The most recent N step_*.pt files
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return

    # Don't touch these special checkpoints
    protected = {'best.pt', 'latest.pt'}

    # Find all step checkpoints and sort by step number
    step_checkpoints = []
    for f in ckpt_dir.glob("step_*.pt"):
        try:
            step_num = int(f.stem.split('_')[1])
            step_checkpoints.append((step_num, f))
        except (ValueError, IndexError):
            continue

    # Sort by step descending, keep newest N, delete the rest
    step_checkpoints.sort(reverse=True)
    for _, f in step_checkpoints[keep_n:]:
        if f.name not in protected:
            f.unlink()
            print(f"  Pruned old checkpoint: {f.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: WordLSTM,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute average validation loss and perplexity over the validation set.

    Args:
        model: The model to evaluate.
        loader: DataLoader yielding (x, y) batches.
        device: Device to run on.
        max_batches: If set, evaluate only the first N batches (for speed).

    Returns:
        Dict with 'loss' and 'perplexity' keys.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        # Count non-padding tokens for proper averaging
        # (sum reduction gives total loss; divide by token count for avg)
        n_tokens = (y != 0).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)

    return {
        'loss':       avg_loss,
        'perplexity': perplexity(avg_loss),
        'tokens':     total_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config: TrainingConfig, resume_from: Optional[str] = None):
    """
    Main training function. Sets up everything and runs the training loop.

    Structure:
      1. Setup: device, seed, directories, logging
      2. Data: load and create loaders
      3. Model: build and move to device
      4. Optimizer + scheduler
      5. Resume (if requested)
      6. Training loop: train step → eval → checkpoint → sample
      7. Early stopping: stops training if validation loss plateaus
      8. Final test evaluation
    """

    # ── Setup ───────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Seed everything for reproducibility
    seed_everything(config.seed)

    # Directories
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "training_log.jsonl"

    logger = TrainingLogger(log_path)
    logger.log("config", **config.to_dict())

    # ── Data ────────────────────────────────────────────────────────────
    print("\nLoading data...")
    if Path(config.vocab_path).exists():
        # Reuse existing vocab (saves time on re-runs)
        vocab = Vocabulary.load(config.vocab_path)
        # Reload raw text to re-create the dataset splits
        with open(config.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        dataset = TextDataset(
            texts=[text], vocab=vocab, min_freq=config.min_freq,
            val_ratio=config.val_ratio, test_ratio=config.test_ratio,
        )
    else:
        # Build vocab from scratch
        text = Path(config.data_path).read_text(encoding='utf-8')
        dataset = TextDataset(
            texts=[text], min_freq=config.min_freq,
            val_ratio=config.val_ratio, test_ratio=config.test_ratio,
        )
        dataset.save_vocab(config.vocab_path)

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        seq_length=config.seq_length,
        batch_size=config.batch_size,
        train_samples_per_epoch=len(dataset.train_data) // config.seq_length,
    )

    logger.log(
        "data",
        vocab_size=len(dataset.vocab),
        train_tokens=len(dataset.train_data),
        val_tokens=len(dataset.val_data),
        test_tokens=len(dataset.test_data),
    )

    # ── Model ───────────────────────────────────────────────────────────
    model_config = ModelConfig(
        vocab_size=len(dataset.vocab),
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout_emb=config.dropout_emb,
        dropout_in=config.dropout_in,
        dropout_out=config.dropout_out,
        tie_weights=config.tie_weights,
    )
    model = WordLSTM(model_config).to(device)

    n_params = model.num_parameters()
    logger.log("model", n_parameters=n_params, **model.parameter_breakdown()['embedding'])
    print(f"Model parameters: {n_params:,}")

    # ── Optimizer + Scheduler ───────────────────────────────────────────
    # AdamW = Adam with proper decoupled weight decay.
    # Regular Adam with weight_decay applies L2 regularization to the
    # gradient, which interacts badly with the adaptive learning rates.
    # AdamW separates them cleanly.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # CosineAnnealingLR: LR follows a cosine curve for T_max epochs,
    # annealing from initial_lr to eta_min. Clean and predictable.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.scheduler_t_max,
        eta_min=config.scheduler_eta_min,
    )

    # Mixed precision scaler (only used if use_amp and CUDA available)
    use_amp = config.use_amp and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    # ── Resume ──────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')    
    patience_counter = 0
    early_stopping_triggered = False
    if resume_from and Path(resume_from).exists():
        print(f"\nResuming from {resume_from}...")
        meta = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        start_epoch = meta['epoch']
        global_step = meta['global_step']
        best_val_loss = meta['best_val_loss']
        logger.log("resume", from_epoch=start_epoch, from_step=global_step)

    # ── Training Loop ───────────────────────────────────────────────────
    print("\nStarting training...")
    print(f"Epochs: {config.epochs}, Effective batch: "
 f"{config.batch_size * config.accumulation_steps}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start = time.time()

        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Forward pass (with optional mixed precision)
            # autocast wraps ops that benefit from FP16/BF16 (matmuls,
            # convolutions) while keeping others in FP32 (loss, softmax).
            with autocast(device_type=device.type, enabled=use_amp):
                logits, _ = model(x)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                # Normalize for gradient accumulation
                loss = loss / config.accumulation_steps

            # Backward
            scaler.scale(loss).backward()

            # Step every accumulation_steps mini-batches
            if (batch_idx + 1) % config.accumulation_steps == 0:
                # Unscale before clipping (scaler scales gradients)
                scaler.unscale_(optimizer)

                # CRITICAL for RNNs: gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.grad_clip,
                )

                # Optimizer step                
                scaler.step(optimizer)
                scaler.update()

                global_step += 1

                # Track loss
                n_tokens = (y != 0).sum().item()
                epoch_loss += loss.item() * config.accumulation_steps * n_tokens
                epoch_tokens += n_tokens

                # ── Periodic evaluation ──────────────────────────────
                if global_step % config.eval_every_n_steps == 0:
                    metrics = evaluate(model, val_loader, device)
                    logger.log(
                        "eval",
                        step=global_step,
                        **metrics,
                        grad_norm=grad_norm.item(),
                        lr=optimizer.param_groups[0]['lr'],
                    )

                    # Track best
                    if metrics['loss'] < best_val_loss:
                        best_val_loss = metrics['loss']
                        save_checkpoint(
                            ckpt_dir / "best.pt",
                            model, optimizer, scheduler,
                            config, model_config,
                            epoch, global_step, best_val_loss,
                            metrics['loss'],
                        )

                    # Back to training mode!
                    model.train()

                # ── Periodic text generation ─────────────────────────
                if global_step % config.sample_every_n_steps == 0:
                    model.eval()
                    sample = generate_sample(
                        model, dataset.vocab,
                        prompt=config.sample_prompt,
                        n_tokens=config.sample_tokens,
                        device=device,
                    )
                    logger.log("sample", step=global_step, text=sample)
                    model.train()

                # ── Periodic checkpoint ──────────────────────────────
                if global_step % config.save_every_n_steps == 0:
                    save_checkpoint(
                        ckpt_dir / "latest.pt",
                        model, optimizer, scheduler,
                        config, model_config,
                        epoch, global_step, best_val_loss,
                        metrics['loss'] if 'metrics' in dir() else 0.0,
                    )
                    save_checkpoint(
                        ckpt_dir / f"step_{global_step}.pt",
                        model, optimizer, scheduler,
                        config, model_config,
                        epoch, global_step, best_val_loss,
                        metrics['loss'] if 'metrics' in dir() else 0.0,
                    )
                    prune_old_checkpoints(
                        str(ckpt_dir),
                        config.keep_last_n_checkpoints,
                    )

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
        epoch_time = time.time() - epoch_start
        logger.log(
            "epoch",
            epoch=epoch,
            train_loss=avg_epoch_loss,
            train_perplexity=perplexity(avg_epoch_loss),
            time_s=epoch_time,
 )

        # Step scheduler at end of epoch
        scheduler.step()

        # Validation at end of epoch
        val_metrics = evaluate(model, val_loader, device)
        logger.log("val_end", epoch=epoch, **val_metrics)

        # Early stopping logic: monitor validation loss and stop training
        # if no improvement is observed for `early_stopping_patience` epochs.
        # This saves compute and prevents overfitting on validation plateau.
        if val_metrics['loss'] < best_val_loss - config.early_stopping_min_delta:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            save_checkpoint(
                ckpt_dir / "best.pt",
                model, optimizer, scheduler,
                config, model_config,
                epoch, global_step, best_val_loss,
                val_metrics['loss'],
            )
            logger.log("best_checkpoint", epoch=epoch, val_loss=best_val_loss)
        else:
            patience_counter += 1
            logger.log("patience", epoch=epoch, patience=patience_counter, best_loss=best_val_loss)
            if patience_counter >= config.early_stopping_patience:
                logger.log(
                    "early_stopping",
                    epoch=epoch,
                    reason=f"no improvement for {config.early_stopping_patience} epochs",
                    best_val_loss=best_val_loss,
                )
                early_stopping_triggered = True
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break

        # Always save latest
        save_checkpoint(
            ckpt_dir / "latest.pt",
            model, optimizer, scheduler,
            config, model_config,
            epoch, global_step, best_val_loss,
            val_metrics['loss'],
        )

    # ── Final test evaluation ───────────────────────────────────────────
    print("\nFinal test evaluation...")
    test_metrics = evaluate(model, test_loader, device)
    logger.log("test_final", **test_metrics)
    print(f"Test perplexity: {test_metrics['perplexity']:.2f}")

    logger.close()
    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 7. Sample generation helper
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample(
    model: WordLSTM,
    vocab: Vocabulary,
    prompt: str,
    n_tokens: int,
    device: torch.device,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """Generate a sample continuation for logging during training."""
    model.eval()

    prompt_ids = vocab.encode(prompt)
    if not prompt_ids:
        return f"[empty prompt]"

    prompt_tensor = torch.tensor([prompt_ids], device=device)
    generated = model.generate(
        prompt_ids=prompt_tensor,
        max_new_tokens=n_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_idx=vocab.eos_idx,
        device=device,
    )
    return vocab.decode(generated[0].tolist())


# ─────────────────────────────────────────────────────────────────────────────
# 8. Utilities
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    """Seed all RNGs for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make CUDA deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> TrainingConfig:
    """Parse command-line arguments. Defaults can be overridden."""
    parser = argparse.ArgumentParser(description="Train LSTM language model")

    # Use argparse to allow overrides but keep defaults in TrainingConfig
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float, dest='learning_rate')
    parser.add_argument('--seq_length', type=int)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--patience', type=int, dest='early_stopping_patience')
    parser.add_argument('--t_max', type=int, dest='scheduler_t_max')

    args = parser.parse_args()
    config = TrainingConfig()

    # Apply overrides
    for k, v in vars(args).items():
        if v is not None and hasattr(config, k):
            setattr(config, k, v)
    if args.no_amp:
        config.use_amp = False

    return config, args.resume


if __name__ == "__main__":
    config, resume = parse_args()
    train(config, resume_from=resume)
