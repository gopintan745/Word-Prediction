"""
model.py
========
LSTM language model with best-practice design choices:

 - Tied embeddings (when dimensions allow)
  - Dropout on embeddings, inputs, and outputs (AWD-LSTM style)
  - Forget gate bias = 1 at initialization
  - Padding index zeroed in embeddings
  - Support for weight-tied projection with optional bias

This module exposes:
  WordLSTM         — the core model
  ModelConfig — dataclass for clean configuration
  init_lstm_weights — standalone weight init function
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Conda (MKL) and PyTorch both ship Intel OpenMP on Windows.
# Without this, importing torch can abort with OMP Error #15.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    Centralized configuration for the LSTM model.

    Using a dataclass instead of scattered kwargs makes it easy to:
      - Pass around the config as a single object
      - Serialize to JSON/YAML for experiment tracking
      - Validate constraints (e.g., tied weights require embed_dim == hidden_size)
    """

    vocab_size: int
    embedding_dim: int = 256
    hidden_size: int = 512
    num_layers: int = 2

    # Regularization
    dropout_emb: float = 0.1      # embedding dropout
    dropout_in: float = 0.3      # LSTM input dropout
    dropout_out: float = 0.3     # LSTM output dropout
    dropout_rec: float = 0.0     # recurrent dropout (use with care)

    # Weight tying
    tie_weights: bool = True
    output_bias: bool = False     # bias on tied projection (rarely useful)

    # Initialization
    init_scale: float = 0.1      # uniform init range for embeddings    
    
    def __post_init__(self):
        # If tying is requested, embedding and hidden sizes must match.
        # Otherwise you'd need a separate adapter, which adds complexity.
        if self.tie_weights and self.embedding_dim != self.hidden_size:
            raise ValueError(
                f"tie_weights=True requires embedding_dim ({self.embedding_dim}) "
                f"== hidden_size ({self.hidden_size}). "
                f"Either set tie_weights=False or match the dimensions."
            )

 # Sanity checks        
        assert self.num_layers >= 1, "Need at least one LSTM layer"
        assert 0 <= self.dropout_emb < 1
        assert 0 <= self.dropout_in < 1
        assert 0 <= self.dropout_out < 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Weight Initialization
# ─────────────────────────────────────────────────────────────────────────────

def init_lstm_weights(lstm: nn.LSTM):
    """
    Apply LSTM-specific initialization to an existing nn.LSTM module.

    Strategy:
      - Recurrent weights (W_hh): orthogonal init (helps gradient flow)
      - Input weights (W_ih): Xavier uniform
      - Forget gate bias: 1.0 (encourage memory retention)
      - Other gate biases: 0.0

    nn.LSTM stores weights as (W_ii, W_if, W_ig, W_io) — 4 gates, each
    of shape (4*hidden_size, input_size) for W_ih and (4*hidden_size,
    hidden_size) for W_hh. PyTorch concatenates them into one tensor per
    layer for efficiency, so we slice accordingly.
    """
    for names in lstm._all_weights:
        # names looks like ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']
        for name in names:
            param = getattr(lstm, name)
            if 'weight_ih' in name:
                # Input-to-hidden: Xavier uniform (good general default)
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-to-hidden: orthogonal (preserves gradient norm)
                # Orthogonal init is critical for RNN stability — it
                # keeps the Jacobian's singular values close to 1.
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Reset all biases to 0 first
                nn.init.zeros_(param)
                # Then bump the forget gate bias to 1
                # The 4 gates are laid out as [i, f, g, o], each of size hidden_size
                hidden_size = lstm.hidden_size
                # PyTorch stacks ih and hh biases separately, but each has
                # the same layout: gates indexed [0:H, H:2H, 2H:3H, 3H:4H]
                param.data[hidden_size:2 * hidden_size].fill_(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. The WordLSTM Model
# ─────────────────────────────────────────────────────────────────────────────

class WordLSTM(nn.Module):
    """
    Word-level LSTM language model with weight tying and AWD-style dropout.

    Forward pass:
        input_ids → embedding (with dropout)
 → LSTM (stacked layers)
                  → output dropout
                  → tied projection to vocab
                  → softmax-ready logits

    Outputs:
        logits: shape (batch, seq_length, vocab_size)
        hidden: final (h, c) state, useful for generation continuation

    Generation helper methods (generate, sample_with_temperature) are
    included so the same model class handles training and inference.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config       
        self.vocab_size = config.vocab_size

        # ── Embedding layer ────────────────────────────────────────────
        # padding_idx=0 means token0 (<pad>) always has zero vector AND
        # zero gradient. This prevents<pad> from contributing to loss        # when masked via ignore_index=0 in cross-entropy.
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=0,
        )

        # ── LSTM ───────────────────────────────────────────────────────
        # batch_first=True: input shape is (batch, seq, feature), which        # is more intuitive than PyTorch's default (seq, batch, feature).
        #
        # dropout is applied to LSTM *inputs and outputs* between layers,
        # but NOT to recurrent transitions. This is the standard safe pattern.
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout_in if config.num_layers > 1 else 0.0,
        )

        # ── Dropout layers (separate so we can apply at different points) ─
        self.dropout_emb = nn.Dropout(config.dropout_emb)
        self.dropout_in = nn.Dropout(config.dropout_in)
        self.dropout_out = nn.Dropout(config.dropout_out)

        # ── Output projection ──────────────────────────────────────────
        if config.tie_weights:
            # We share weights with the embedding matrix.
            # The forward pass computes h @ embedding.weight.T directly.
            # No nn.Linear needed; this saves parameters and improves results.
            if config.output_bias:
                # Optional small bias per vocab token. Usually omitted because
                # any token-specific bias can be absorbed into the embedding.
                self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))
            else:
                self.output_bias = None
            self.classifier = None
        else:
            # Separate projection: more parameters but allows embedding_dim ≠ hidden_size.
            self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
            self.output_bias = None

        # ── Apply weight initialization ────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Initialize all parameters."""
        # Embeddings: uniform init with a small range
        nn.init.uniform_(
            self.embedding.weight,
            -self.config.init_scale,
            self.config.init_scale,
        )
        # Zero out the padding row — it's a placeholder, never used
        with torch.no_grad():
            self.embedding.weight[0].zero_()

        # LSTM weights: forget bias = 1, orthogonal recurrent, Xavier input
        init_lstm_weights(self.lstm)

        # Classifier (only if not tied)
        if self.classifier is not None:
            nn.init.xavier_uniform_(self.classifier.weight)
            if self.classifier.bias is not None:
                nn.init.zeros_(self.classifier.bias)

    # ─────────────────────────────────────────────────────────────────────
    # 4. Forward pass
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input token IDs, shape (batch, seq_length)
            hidden: Optional (h, c) initial state. Each is shape
 (num_layers, batch, hidden_size). If None, starts at zero.

        Returns:
            logits: shape (batch, seq_length, vocab_size)
            hidden: final (h, c) state after the full sequence
        """
        # Step 1: Embed tokens → dense vectors
        # Shape: (batch, seq_length, embedding_dim)
        embedded = self.embedding(x)

        # Step 2: Dropout on embeddings (regularization, see AWD-LSTM)
        embedded = self.dropout_emb(embedded)

        # Step 3: Dropout on LSTM inputs (between embedding and LSTM)
        #embedded = self.dropout_in(embedded)

        # Step 4: Run the LSTM
        # output shape: (batch, seq_length, hidden_size)
        # hidden = (h_n, c_n), each shape (num_layers, batch, hidden_size)
        output, hidden = self.lstm(embedded, hidden)

        # Step 5: Dropout on LSTM outputs (between LSTM and projection)
        output = self.dropout_out(output)

        # Step 6: Project to vocabulary size
        if self.classifier is not None:
            logits = self.classifier(output)
        else:
            # Tied projection: use embedding matrix transposed
            # Shape: (batch, seq, hidden) @ (hidden, vocab) → (batch, seq, vocab)
            logits = F.linear(output, self.embedding.weight, self.output_bias)

        return logits, hidden

    # ─────────────────────────────────────────────────────────────────────
    # 5. Hidden state helpers
    # ─────────────────────────────────────────────────────────────────────

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state to zeros. Use at the start of generation,
        or at the start of training (for stateful training across chunks).
        """
        h = torch.zeros(
            self.config.num_layers,
            batch_size,
            self.config.hidden_size,
            device=device,
        )
        c = torch.zeros_like(h)
        return (h, c)

    def detach_hidden(
        self,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detach hidden state from computation graph.

        Critical for Truncated BPTT: prevents gradients from flowing
        across chunk boundaries. Without this, you'd backprop through
        the entire training set every step, which is impossibly slow
        and causes memory blowup.

        Args:
            hidden: (h, c) tuple of shape (num_layers, batch, hidden_size)

        Returns:
            Same shape but detached — `.detach()` returns a tensor that
            shares data but doesn't track gradients.
        """
        h, c = hidden
        return (h.detach(), c.detach())

    # ─────────────────────────────────────────────────────────────────────
    # 6. Text generation (used by the Streamlit app)
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_idx: Optional[int] = None,
        device: torch.device = torch.device('cpu'),
    ) -> torch.Tensor:
        """
        Autoregressive text generation from a prompt.

        Generation loop:
          1. Forward pass on current context → logits over next token
          2. Apply temperature scaling (controls randomness)
          3. Optionally filter to top-k or nucleus (top-p) tokens
          4. Sample from resulting distribution
          5. Append sampled token to context, repeat

        Args:
            prompt_ids: shape (1, prompt_length) — already on `device`
            max_new_tokens: how many tokens to generate            
            temperature: <1 makes it more confident, >1 more random
            top_k: keep only the top-k most likely tokens
            top_p: keep smallest set of tokens with cumulative prob ≥ top_p
            eos_idx: stop early if <eos> is generated
            device: where to run inference

        Returns:
            Tensor of shape (1, prompt_length + new_tokens) with full sequence
        """
        self.eval()
        ids = prompt_ids.clone()
        hidden = self.init_hidden(batch_size=1, device=device)

        for _ in range(max_new_tokens):
            # Forward pass on the full current context.
            # In production, you'd cache the hidden state and only run the
            # LSTM on the new token — but for simplicity we recompute.
            logits, hidden = self.forward(ids, hidden)

            # Take the prediction for the LAST position (next-token prediction)
            next_logits = logits[0, -1, :]  # shape: (vocab_size,)

            # Temperature: divide logits by T before softmax.
            # T → 0: deterministic argmax
            # T → ∞: uniform random
            # T = 1: standard softmax
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Top-k filtering: keep only the k highest-prob tokens
            if top_k is not None:
                # Find the k-th largest logit value; zero out everything below
                kth_values, _ = torch.topk(next_logits, top_k)
                kth_value = kth_values[-1]
                next_logits = torch.where(
                    next_logits < kth_value,
                    torch.full_like(next_logits, float('-inf')),
                    next_logits,
                )

            # Top-p (nucleus) filtering: keep smallest set with cumsum ≥ p
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Create a mask for tokens to remove (cumulative prob > p)
                sorted_mask = cumulative_probs > top_p
                # Always keep at least the top token
                sorted_mask[0] = False

                # Scatter mask back to original indices
                mask = torch.zeros_like(next_logits, dtype=torch.bool)
                mask.scatter_(0, sorted_idx, sorted_mask)
                next_logits = next_logits.masked_fill(mask, float('-inf'))

            # Convert to probabilities and sample
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # shape: (1,)

            # Append the new token
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)

            # Early stopping on <eos>
            if eos_idx is not None and next_id.item() == eos_idx:
                break

        return ids # ─────────────────────────────────────────────────────────────────────
    # 7. Parameter counting (useful for tracking model size)
    # ─────────────────────────────────────────────────────────────────────

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Return the number of (trainable) parameters."""
        return sum(
            p.numel() for p in self.parameters()
            if (not only_trainable) or p.requires_grad
        )

    def parameter_breakdown(self) -> dict:
        """Print which components contribute how many parameters."""
        breakdown = {
            'embedding': self.embedding.weight.numel(),
            'lstm':     sum(p.numel() for p in self.lstm.parameters()),
            'classifier': self.classifier.weight.numel() if self.classifier else 0,
            'output_bias': self.output_bias.numel() if self.output_bias is not None else 0,
        }
        total = sum(breakdown.values())
        return {k: {'count': v, 'pct': 100 * v / total} for k, v in breakdown.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 8. Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Run `python src/model.py` for a quick architecture sanity check."""

    print("=" * 60)
    print("LSTM Model — smoke test")
    print("=" * 60)

    #1. Small config for fast testing
    config = ModelConfig(
        vocab_size=1000,
        embedding_dim=128,
        hidden_size=128,
        num_layers=2,
        dropout_emb=0.1,
        dropout_in=0.3,
        dropout_out=0.3,
        tie_weights=True,
    )
    model = WordLSTM(config)
    model.eval()  # disable dropout for testing

    # 2. Parameter breakdown
    print(f"\nTotal parameters: {model.num_parameters():,}")
    print("\nParameter breakdown:")
    for name, info in model.parameter_breakdown().items():
        print(f"  {name:12s} {info['count']:>10,}  ({info['pct']:5.2f}%)")

    # 3. Forward pass with random input
    batch_size, seq_length = 4, 16
    x = torch.randint(0, 1000, (batch_size, seq_length))
    logits, hidden = model(x)

    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Hidden h shape: {hidden[0].shape}")
    print(f"  Hidden c shape: {hidden[1].shape}")

    # 4. Verify hidden state carries forward correctly
    print("\n4. Hidden state detach test")

    # Detach should preserve values but break gradient tracking
    h_orig = hidden[0]
    c_orig = hidden[1]

    h_det, c_det = model.detach_hidden((h_orig, c_orig))

    # Values should match exactly
    assert torch.allclose(h_orig, h_det), "Detach modified h values!"
    assert torch.allclose(c_orig, c_det), "Detach modified c values!"

    # Gradient tracking should be off
    assert not h_det.requires_grad, "h_det.requires_grad should be False"
    assert not c_det.requires_grad, "c_det.requires_grad should be False"
    assert h_orig.requires_grad, "h_orig.requires_grad should be True"

    # 5. Generation smoke test
    prompt = torch.tensor([[5, 23, 47, 99]])  # (1, 4)
    generated = model.generate(
        prompt_ids=prompt,
        max_new_tokens=10,
        temperature=1.0,
        top_k=5,
    )
    print(f"\n  Generated shape: {generated.shape}")
    print(f"  Generated IDs:   {generated[0].tolist()}")

    print("\n6. Eval mode determinism test")

    model.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        out1, hid1 = model(x)
        torch.manual_seed(0)
        out2, hid2 = model(x)

    assert torch.allclose(out1, out2), "Same seed should give same output!"
    assert torch.allclose(hid1[0], hid2[0]), "Hidden state not deterministic!"
    print("   ✓ Forward pass is deterministic in eval mode")

 # ─────────────────────────────────────────────────────────────────
# 7. Gradient flow sanity check
# ─────────────────────────────────────────────────────────────────

    print("\n7. Gradient flow test")

    model.train()  # re-enable dropout
    x_train = torch.randint(0, 1000, (2, 8))
    y_train = torch.randint(0, 1000, (2, 8))

    logits, _ = model(x_train)
    loss = F.cross_entropy(logits.reshape(-1, 1000), y_train.reshape(-1))
    loss.backward()

    # Every trainable parameter should have a non-None gradient
    missing_grads = [
        n for n, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert len(missing_grads) == 0, f"No gradient for: {missing_grads}"

    # No NaN gradients (NaN != NaN, so this trick catches them)
    nan_grads = [
        n for n, p in model.named_parameters()
        if p.grad is not None and torch.isnan(p.grad).any()
    ]
    assert len(nan_grads) == 0, f"NaN gradients in: {nan_grads}"

    # No Inf gradients
    inf_grads = [
        n for n, p in model.named_parameters()
        if p.grad is not None and torch.isinf(p.grad).any()
    ]
    assert len(inf_grads) == 0, f"Inf gradients in: {inf_grads}"

    print(f"   ✓ Loss: {loss.item():.4f}")
    print(f"   ✓ All {len(list(model.parameters()))} parameter groups have gradients")
    print(f"   ✓ No NaN or Inf gradients")

    # Bonus: print gradient norms for debugging
    print("\n   Gradient norm summary:")
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"     {name:30s} {p.grad.norm().item():.6f}")

    print("\n" + "=" * 60)
    print("All smoke tests passed!")
    print("=" * 60)
