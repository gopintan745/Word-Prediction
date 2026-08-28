"""
evaluate.py
===========
Comprehensive evaluation for trained WordLSTM language models.

Usage:
    # Evaluate a checkpoint
    python src/evaluate.py --checkpoint checkpoints/best.pt
    
    # Evaluate with multiple samples
    python src/evaluate.py --checkpoint checkpoints/best.pt --num_samples 20
    
    # Compare multiple checkpoints
    python src/evaluate.py --compare checkpoints/best.pt checkpoints/latest.pt

Features:
    - Perplexity and bits-per-word
    - Top-k accuracy (next-token prediction)
    - Diverse text generation with multiple sampling strategies
    - Per-position loss analysis (early vs. late in sequence)
    - Save results to JSON for experiment tracking
"""

import os
# Conda (MKL) and PyTorch both ship Intel OpenMP on Windows.
# Without this, importing torch can abort with OMP Error #15.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (
    TextDataset, Vocabulary, create_dataloaders,
)
from model import WordLSTM, ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Results container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """
    Container for all evaluation metrics and outputs.
    
    Storing everything in a dataclass makes it easy to:
      - Save to JSON (just `asdict()`)
      - Compare across runs
      - Pass to other scripts (Streamlit, reporting)
    """
    # Identification
    checkpoint_path: str
    
    # Core metrics
    loss: float
    perplexity: float
    bits_per_word: float
    
    # Top-k accuracy (next-token prediction)
    top1_accuracy: float
    top5_accuracy: float
    top10_accuracy: float
    
    # Per-position analysis
    early_position_loss: float    # first 25% of sequence
    middle_position_loss: float    # middle 50% of sequence
    late_position_loss: float     # last 25% of sequence
    
    # Token-level breakdown
    rare_word_loss: float         # loss on <unk> candidates
    common_word_loss: float # loss on high-frequency words
    
    # Samples
    samples: List[Dict] # generated samples with metadata
    
    # Metadata
    n_tokens_evaluated: int
    n_batches_evaluated: int
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def print_report(self):
        """Pretty-print the results to console."""
        print("\n" + "=" * 70)
        print(f"Evaluation Report: {self.checkpoint_path}")
        print("=" * 70)
        print(f"\n Core Metrics:")
        print(f"   Loss (cross-entropy): {self.loss:.4f}")
        print(f"   Perplexity:           {self.perplexity:.2f}")
        print(f"   Bits-per-word:        {self.bits_per_word:.4f}")
        print(f"\n Top-k Accuracy:")
        print(f"   Top-1:   {self.top1_accuracy*100:.2f}%")
        print(f"   Top-5:   {self.top5_accuracy*100:.2f}%")
        print(f"   Top-10:  {self.top10_accuracy*100:.2f}%")
        print(f"\n Per-Position Loss:")
        print(f"   Early: {self.early_position_loss:.4f}")
        print(f"   Middle: {self.middle_position_loss:.4f}")
        print(f"   Late:   {self.late_position_loss:.4f}")
        if hasattr(self, 'early_position_loss') and hasattr(self, 'late_position_loss'):
            trend = 'worse' if self.late_position_loss > self.early_position_loss else 'better'
            print(f"   Trend:  {trend}")
        print(f"\n Token-Type Loss:")
        print(f"   Rare words:   {self.rare_word_loss:.4f}")
        print(f"   Common words: {self.common_word_loss:.4f}")
        print(f"\n Data:")
        print(f"   Tokens evaluated: {self.n_tokens_evaluated:,}")
        print(f"   Batches evaluated: {self.n_batches_evaluated}")
        print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device = torch.device('cpu'),
) -> Tuple[WordLSTM, ModelConfig, dict]:
    """
    Load a model from a checkpoint file.
    
    Returns the model, its config, and the full checkpoint metadata.
    The checkpoint stores both the model_config (for rebuilding the    architecture) and training metadata (epoch, step, etc.).
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False    )
    
    # Rebuild the model architecture from the saved config
    cfg_dict = checkpoint['model_config']
    model_config = ModelConfig(**cfg_dict)
    model = WordLSTM(model_config)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # CRITICAL: disable dropout for evaluation
    
    metadata = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'global_step': checkpoint.get('global_step', 'unknown'),
        'best_val_loss': checkpoint.get('best_val_loss', float('nan')),
    }
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {metadata['epoch']}, Step: {metadata['global_step']}")
    print(f"  Best val loss: {metadata['best_val_loss']:.4f}")
    
    return model, model_config, metadata


# ─────────────────────────────────────────────────────────────────────────────
# 3. Core metric computations
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_loss_and_perplexity(
    model: WordLSTM,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute cross-entropy loss and perplexity on a dataset.
    
    Why max_batches? For quick checks during HPO, you might only need
    ~100 batches. For final evaluation, set None to use the full set.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        
        # reduction='sum' gives total loss, we divide by token count
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=0,  # ignore padding
            reduction='sum',
        )
        
        n_tokens = (y != 0).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens
    
    if total_tokens == 0:
        raise ValueError("No tokens evaluated! Check the data loader.")
    
    avg_loss = total_loss / total_tokens
    return {
        'loss': avg_loss,
        'perplexity': math.exp(avg_loss),
        'bits_per_word': avg_loss / math.log(2),  # convert nats to bits
        'tokens': total_tokens,
 'batches': i + 1,
    }


@torch.no_grad()
def compute_top_k_accuracy(
    model: WordLSTM,
    loader: DataLoader,
    device: torch.device,
    k_values: List[int] = [1, 5, 10],
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute top-k accuracy for next-token prediction.
    
    "Top-k accuracy" = fraction of times the true next token is among
    the model's top-k predictions. This is a more forgiving metric than
    perplexity because it doesn't care about exact probability.
    
    Top-1 = exact match rate (rare for LMs because vocab is huge)
    Top-5 = correct in top 5 guesses
    Top-10 = correct in top 10 guesses    A good LSTM on Shakespeare might achieve:
      Top-1:  ~15-20%
      Top-5:  ~40-50%
      Top-10: ~55-65%
    """
    model.eval()
    
    # Track per-k correct predictions and total predictions
    correct_counts = {k: 0 for k in k_values}
    total = 0
    
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        
        # Get top-k predictions: shape (batch, seq, k)
        # torch.topk returns (values, indices)
        max_k = max(k_values)
        _, top_k_pred = torch.topk(logits, k=max_k, dim=-1)
        
        # Check if true tokens are in top-k
        # y: (batch, seq), top_k_pred: (batch, seq, k)
        # We need: y_expanded shape (batch, seq, k) to compare
        y_expanded = y.unsqueeze(-1)  # (batch, seq, 1)
 # Broadcast: (batch, seq, 1) vs (batch, seq, k) → (batch, seq, k)
        is_in_topk = (top_k_pred == y_expanded)  # bool tensor
        
        # Mask out padding tokens (don't count them)
        non_pad = (y != 0).unsqueeze(-1)  # (batch, seq, 1)
        is_in_topk = is_in_topk & non_pad
        
        for k in k_values:
            # Top-k is correct if true token is in top-k predictions
            correct_counts[k] += is_in_topk[..., :k].any(dim=-1).sum().item()
        
        total += non_pad.sum().item()
    
    return {
        f'top{k}_accuracy': correct_counts[k] / max(total, 1)
        for k in k_values
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-position loss analysis
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_per_position_loss(
    model: WordLSTM,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute loss separately for early vs. late positions in sequences.
    
    This is diagnostic: it tells you whether the model is struggling
    with short or long contexts.
    
    Early positions (first 25%) have little context → harder to predict
    Late positions (last 25%) have full context → should be easier
    If late > early, something's wrong (maybe overfitting to short deps).
    """
    model.eval()
    
    # We accumulate loss per position relative to sequence length
    position_losses = {}  # bucket → [losses]
    
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        
        seq_len = x.size(1)
        # Per-token losses: (batch, seq)
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=0,
            reduction='none',
        ).reshape(x.size(0), seq_len)
        
        # Define buckets by relative position
        for start_frac, end_frac, name in [
            (0.0,  0.25, 'early'),
            (0.25, 0.75, 'middle'),
            (0.75, 1.0,  'late'),
        ]:
            start = int(seq_len * start_frac)
            end = int(seq_len * end_frac)
            # Average loss in this position range, ignoring padding
            mask = (y[:, start:end] != 0).float()
            mean_loss = (
                (token_losses[:, start:end] * mask).sum() / mask.sum().clamp(min=1)
            ).item()
            position_losses.setdefault(name, []).append(mean_loss)
    
    # Average across batches
    return {
        f'{name}_position_loss': sum(losses) / len(losses)
        for name, losses in position_losses.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sample generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(
    model: WordLSTM,
    vocab: Vocabulary,
    prompts: List[str],
    device: torch.device,
    n_tokens: int = 50,
    sampling_configs: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Generate text samples with multiple sampling strategies.
    
    Sampling strategies matter a lot for perceived quality:
      - Greedy (T=0): boring, repetitive
      - High T (1.5): creative but incoherent
      - Top-k with k=40: balanced      - Nucleus (p=0.9): quality-preserving
    
    By trying multiple configs, you can see which produces best output.
    """
    if sampling_configs is None:
        sampling_configs = [
            {'name': 'greedy',       'temperature': 0.0, 'top_k': None,  'top_p': None},
            {'name': 'low_temp',     'temperature': 0.5, 'top_k':20,    'top_p': None},
            {'name': 'medium_temp',  'temperature': 0.8, 'top_k': 40,    'top_p':0.9},
            {'name': 'high_temp',    'temperature': 1.2, 'top_k': None,  'top_p': 0.95},
        ]
    
    samples = []
    for prompt in prompts:
        prompt_ids = vocab.encode(prompt)
        if not prompt_ids:
            continue        
        prompt_tensor = torch.tensor([prompt_ids], device=device)
        
        for config in sampling_configs:
            # Special case: greedy = temperature 0
            if config.get('temperature', 1.0) == 0.0:
                generated_ids = greedy_generate(
                    model, prompt_tensor, n_tokens, device )
            else:
                generated_ids = model.generate(
                    prompt_ids=prompt_tensor,
                    max_new_tokens=n_tokens,
                    temperature=config.get('temperature', 1.0),
                    top_k=config.get('top_k'),
                    top_p=config.get('top_p'),
                    eos_idx=vocab.eos_idx,
                    device=device,
                )[0].cpu().tolist()
            
            text = vocab.decode(generated_ids)
            samples.append({
                'prompt': prompt,
                'config': config['name'],
                'temperature': config.get('temperature'),
                'top_k': config.get('top_k'),
                'top_p': config.get('top_p'),
                'text': text,
            })
    
    return samples


@torch.no_grad()
def greedy_generate(
    model: WordLSTM,
    prompt_ids: torch.Tensor,
    n_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Greedy decoding: always pick the argmax token.
    
    Bypasses the random sampling in model.generate for deterministic output.
    """
    model.eval()
    ids = prompt_ids.clone()
    hidden = model.init_hidden(batch_size=1, device=device)
    
    for _ in range(n_tokens):
        logits, hidden = model(ids, hidden)
        next_id = logits[0, -1].argmax().item()
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    
    return ids[0]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    checkpoint_path: str,
    dataset: TextDataset,
    config_overrides: Optional[dict] = None,
    prompts: Optional[List[str]] = None,
    n_sample_tokens: int = 50,
    max_batches: Optional[int] = None,
    output_path: Optional[str] = None,
) -> EvaluationResult:
    """
    Full evaluation pipeline for a checkpoint.
    
    Args:
        checkpoint_path: Path to .pt file
        dataset: TextDataset with vocab and splits
        config_overrides:    Optional dict to override model config
                             (rarely needed — checkpoint stores config)
        prompts:             List of prompts for generation. Defaults to a few Shakespeare-style prompts.
        n_sample_tokens:     How many tokens to generate per sample
        max_batches:         Cap on evaluation batches (None = full set)
        output_path:         If set, save results JSON to this path
    
    Returns:
        EvaluationResult dataclass with everything
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Default prompts for generation
    if prompts is None:
        prompts = [
            "to be or not to",
            "ROMEO:",
            "the king",
            "love is",
        ]
    
    # Load model
    model, model_config, metadata = load_model_from_checkpoint(
        checkpoint_path, device
    )
    vocab = dataset.vocab
    
    # Build test loader
    _, _, test_loader = create_dataloaders(
        dataset,
        seq_length=model_config.embedding_dim,  # doesn't matter much for eval
        batch_size=64,
    )
    
    print("\nComputing metrics on test set...")
    
    # Core loss/perplexity
    loss_metrics = compute_loss_and_perplexity(
        model, test_loader, device, max_batches
    )
    
    # Top-k accuracy
    acc_metrics = compute_top_k_accuracy(
        model, test_loader, device, k_values=[1, 5, 10], max_batches=max_batches
    )
    
    # Per-position analysis
    pos_metrics = compute_per_position_loss(
        model, test_loader, device, max_batches
    )
    
    # Sample generation
    print(f"\nGenerating {len(prompts)} × 4 samples...")
    samples = generate_samples(
        model, vocab, prompts, device, n_tokens=n_sample_tokens
    )
    
    # Build result
    result = EvaluationResult(
        checkpoint_path=str(checkpoint_path),
        loss=loss_metrics['loss'],
        perplexity=loss_metrics['perplexity'],
        bits_per_word=loss_metrics['bits_per_word'],
        top1_accuracy=acc_metrics['top1_accuracy'],
        top5_accuracy=acc_metrics['top5_accuracy'],
        top10_accuracy=acc_metrics['top10_accuracy'],
        early_position_loss=pos_metrics['early_position_loss'],
        middle_position_loss=pos_metrics['middle_position_loss'],
        late_position_loss=pos_metrics['late_position_loss'],
        rare_word_loss=0.0,    # placeholder; could implement separately
        common_word_loss=0.0,  # placeholder
        samples=samples,
        n_tokens_evaluated=int(loss_metrics['tokens']),
        n_batches_evaluated=int(loss_metrics['batches']),
    )
    
    # Print to console
    result.print_report()
    
    # Print samples
    print("\n Generated Samples:")
    print("-" * 70)
    for sample in samples:
        prompt_preview = sample['prompt'][:30]
        text_preview = sample['text'][:120]
        print(f"\n [{sample['config']:12s}] T={sample['temperature']:.1f} "
 f"top_k={sample['top_k']} top_p={sample['top_p']}")
        print(f"   Prompt: \"{prompt_preview}...\"")
        print(f"   Output: \"{text_preview}...\"")
    print("-" * 70)
    
    # Save to JSON
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. Compare multiple checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def compare_checkpoints(
    checkpoint_paths: List[str],
    dataset: TextDataset,
    max_batches: Optional[int] = None,
) -> None:
    """
    Side-by-side comparison of multiple checkpoints.
    
    Useful for HPO: see which config produced the best model.
    """
    print("\n" + "=" * 70)
    print(f"Comparing {len(checkpoint_paths)} checkpoints")
    print("=" * 70)
    
    results = []
    for path in checkpoint_paths:
        print(f"\n{'─'*70}")
        print(f"Evaluating: {path}")
        print('─'*70)
        result = evaluate_checkpoint(
            path, dataset, max_batches=max_batches
        )
        results.append(result)
    
    # Summary table
    print("\n\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"{'Checkpoint':<45} {'PPL':>10} {'Top-1':>8} {'Top-5':>8}")
    print("-" * 70)
    for r in results:
        # Truncate long paths
        path_short = r.checkpoint_path[-42:] if len(r.checkpoint_path) > 45 else r.checkpoint_path
        print(f"{path_short:<45} {r.perplexity:>10.2f} "
 f"{r.top1_accuracy*100:>7.2f}% {r.top5_accuracy*100:>7.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LSTM language model")
    parser.add_argument(
        '--checkpoint', '-c',
        required=True,
        help='Path to checkpoint (or comma-separated list for comparison)'
    )
    parser.add_argument(
        '--data', '-d',
        default='data/raw/tinyshakespeare.txt',
        help='Path to text data'
    )
    parser.add_argument(
        '--vocab', '-v',
        default='data/processed/vocab.pkl',
        help='Path to saved vocabulary'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Save results JSON to this path'
    )
    parser.add_argument(
        '--max_batches',
        type=int, default=None,
        help='Cap evaluation to N batches (for quick checks)'
    )
    parser.add_argument(
        '--n_tokens',
        type=int, default=50,
        help='Number of tokens to generate per sample'
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='Treat checkpoint arg as comma-separated list to compare'
    )
    parser.add_argument(
        '--prompts', nargs='+', default=None,
        help='Custom prompts for generation'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Load dataset
    if Path(args.vocab).exists():
        vocab = Vocabulary.load(args.vocab)
        text = Path(args.data).read_text(encoding='utf-8')
        dataset = TextDataset(texts=[text], vocab=vocab)
    else:
        text = Path(args.data).read_text(encoding='utf-8')
        dataset = TextDataset(texts=[text])
    
    # Run evaluation or comparison
    if args.compare or ',' in args.checkpoint:
        paths = [p.strip() for p in args.checkpoint.split(',')]
        compare_checkpoints(paths, dataset, args.max_batches)
    else:
        evaluate_checkpoint(
            args.checkpoint,
            dataset,
            prompts=args.prompts,
            n_sample_tokens=args.n_tokens,
            max_batches=args.max_batches,
            output_path=args.output,
        )
