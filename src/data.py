"""
data.py
=======
Data pipeline for word-level language modeling with LSTMs.

This module handles:
 1. Vocabulary construction (text → integer IDs)
  2. Dataset preparation (loading, splitting, batching)
  3. Batching with the input/target shift trick
  4. Optional packing for variable-length sequences

Key design decisions:
  - Word-level tokenization (simpler, matches project goal)
  - Fixed-window sampling (random offsets per batch) — standard for LSTMs
  - Special tokens: <pad>=0, <unk>=1, <bos>=2, <eos>=3
"""

import re
import pickle
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# 1. Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    """
    Maps words to integer IDs and back.

    Special tokens are reserved at fixed indices:
        0: <pad>  — padding token (ignored in loss)
        1: <unk>  — unknown token (OOV words)
        2: <bos>  — beginning of sequence
        3: <eos>  — end of sequence

    The vocabulary is built from a frequency threshold (`min_freq`) to remove
    rare words that would mostly become <unk>. Optionally capped at `max_size`
    to keep embedding matrices manageable.
    """

    def __init__(
        self,
        texts: List[str],
        min_freq: int = 2,
        max_size: Optional[int] = None,
    ):
        # Step 1: Count all word frequencies across the corpus
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        # Step 2: Reserve special tokens FIRST so their indices are stable.
        # This means anyone loading this vocab gets the same indices.
        self.specials = ['<pad>', '<unk>', '<bos>', '<eos>']
        self.pad_idx = 0
        self.unk_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3

        # Step 3: Filter by frequency, keep most common N
        # `most_common()` already sorts by count descending, so we get
        # frequency-ordered vocabulary after specials.
        word_freqs = counter.most_common()
        if min_freq > 1:
            word_freqs = [(w, c) for w, c in word_freqs if c >= min_freq]
        if max_size:
            word_freqs = word_freqs[: max_size]

        # Step 4: Build the two lookup tables
        # itos = index-to-string (used for decoding back to text)
        # stoi = string-to-index (used for encoding text to IDs)
        words = [w for w, _ in word_freqs]
        self.itos = self.specials + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Split text into words. We use simple whitespace + punctuation splitting.

        For production, you'd use a proper tokenizer (BPE, WordPiece, spaCy).
        For learning purposes, this is transparent and easy to reason about.

        The regex captures:
          - sequences of letters/numbers/apostrophes (e.g., "don't", "LSTM")
          - individual punctuation marks (e.g., ".", ",")
        """
        # Keep contractions and alphanumerics as one token; split punctuation.
        pattern = r"\w+(?:'\w+)*|[^\w\s]"
        return re.findall(pattern, text.lower())

    def encode(self, text: str) -> List[int]:
        """
        Convert text → list of integer IDs.

        Unknown words (not in vocab or below min_freq) map to <unk>=1.
        Empty list returned for empty input.
        """
        return [self.stoi.get(tok, self.unk_idx) for tok in self._tokenize(text)]

    def decode(self, ids: List[int], skip_specials: bool = True) -> str:
        """
        Convert list of integer IDs → text.

        By default, special tokens are omitted from output (cleaner for generation).
        """
        tokens = []
        for i in ids:
            tok = self.itos[i] if i < len(self.itos) else '<unk>'
            if skip_specials and tok in self.specials:
                continue
            tokens.append(tok)
        # Reconstruct with spacing rules: keep punctuation attached to previous word
        text = ""
        for tok in tokens:
            if tok in ',.!?;:)':
                text += tok
            else:
                text += " " + tok
        return text.strip()

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: str):
        """Persist vocab to disk. Use this after building so you don't rebuild every run."""
        with open(path, 'wb') as f:
            pickle.dump({
                'itos': self.itos,
                'stoi': self.stoi,
                'specials': self.specials,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load a saved vocab. We rebuild the object so all methods work."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.itos = data['itos']
        obj.stoi = data['stoi']
        obj.specials = data['specials']
        obj.pad_idx = obj.stoi['<pad>']
        obj.unk_idx = obj.stoi['<unk>']
        obj.bos_idx = obj.stoi['<bos>']
        obj.eos_idx = obj.stoi['<eos>']
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TextDataset:
    """
    Wraps a tokenized corpus with train/val/test splits.

    We tokenize the entire corpus once and store it as a NumPy int32 array.
    NumPy is more memory-efficient than Python lists for millions of integers,
    and PyTorch's DataLoader can yield NumPy arrays directly.

    Splits are sequential (not random):
        [0 ........ val_start ........ test_start ........ end]
                          │ │
 val set          test set
                          └──── train ───┘

    Sequential splits preserve natural text order, which matters for language
    modeling — random shuffling would leak future context into training.
    """

    def __init__(
        self,
        texts: List[str],
        vocab: Optional[Vocabulary] = None,
        min_freq: int = 2,
        max_vocab_size: Optional[int] = None,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
    ):
        # Build vocab if not provided. This step is expensive, so for large
        # corpora you might want to cache it.
        if vocab is None:
            vocab = Vocabulary(texts, min_freq=min_freq, max_size=max_vocab_size)

        self.vocab = vocab

        # Tokenize once, encode to ints, store as NumPy array.
        # Joining all texts with <eos> between them gives the model a clear
        # boundary signal. Each document becomes a coherent sequence, then
        # the <eos> breaks context between documents.
        token_streams = []
        for text in texts:
            ids = vocab.encode(text)
            if ids:
                token_streams.append(ids)
                token_streams.append([vocab.eos_idx])
        all_ids = [tid for stream in token_streams for tid in stream]
        self.data = np.array(all_ids, dtype=np.int64)

        # Sequential splits
        n = len(self.data)
        test_start = int(n * (1 - test_ratio))
        val_start = int(n * (1 - test_ratio - val_ratio))

        self.train_data = self.data[:val_start]
        self.val_data   = self.data[val_start:test_start]
        self.test_data  = self.data[test_start:]

        # Diagnostics — useful for debugging the pipeline
        print(f"[TextDataset] Vocab size: {len(vocab):,}")
        print(f"[TextDataset] Total tokens: {n:,}")
        print(f"[TextDataset] Train: {len(self.train_data):,}  "
              f"Val: {len(self.val_data):,}  Test: {len(self.test_data):,}")

    def save_vocab(self, path: str):
        self.vocab.save(path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Random Window Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RandomWindowDataset(Dataset):
    """
    Yields (input, target) pairs from a corpus using random offset sampling.

    This is the standard approach for language model training because:
      - Each batch sees different positions → better gradient signal
      - No need to track epoch boundaries
      - Natural data augmentation (many ways to slice the same corpus)

    Important: the target is the input shifted right by one position. This is the "input/target shift trick" — one forward pass produces N predictions,
    one for each position, all evaluated against the next-token target.

    Input/Target shape: (batch_size, seq_length)
        Example (seq_length=4):
            input:  [10, 20, 30, 40]
            target: [20, 30, 40, 50] ← shifted by 1
 """

    def __init__(self, data: np.ndarray, seq_length: int, num_samples: int):
        """
        Args:
            data:1-D NumPy array of token IDs.
            seq_length:  Number of tokens per training example (chunk size for TBPTT).
            num_samples: Number of (input, target) pairs to draw per "epoch".
                         Set this large (e.g., len(data) // seq_length * 10)
                         to give the model many chances to see each token.
        """
        # We need seq_length + 1 tokens per sample because we need both
        # the input window AND the shifted target window.
        if len(data) <= seq_length:
            raise ValueError(
                f"Data length ({len(data)}) must be > seq_length ({seq_length})"
            )

        self.data = data
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns one random (input, target) window.

        The `idx` argument is unused because we sample randomly — PyTorch needs
        __getitem__ to accept an index, but we ignore it. This is the
        recommended pattern when each "sample" is itself a random slice.

        We subtract (seq_length + 1) from the maximum offset so we never        sample past the end of the array.
        """
        max_offset = len(self.data) - self.seq_length - 1
        start = np.random.randint(0, max_offset)

        # Input window        
        x = self.data[start : start + self.seq_length]
        # Target window — shifted by 1 (the next-token target)
        y = self.data[start + 1 : start + self.seq_length + 1]

        return (
            torch.from_numpy(x.copy()).long(),
            torch.from_numpy(y.copy()).long(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    dataset: TextDataset,
    seq_length: int = 64,
    batch_size: int = 64,
    num_workers: int = 0,
    train_samples_per_epoch: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build DataLoaders for train/val/test.

    Train uses random window sampling (shuffle-equivalent).
    Val/test use a deterministic sequential pass over the held-out data,
    so metrics are reproducible and don't depend on the random seed.

    Args:
        dataset:                   The TextDataset holding splits.
        seq_length:                Tokens per training example.
        batch_size:                Sequences per batch.
        num_workers:               Parallel data loading workers.
        train_samples_per_epoch:   How many (input, target) windows to draw
                                   per epoch. If None, defaults to ~ one full
                                   pass over the training data (len / seq_length).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Train: random sampling, many samples per epoch
    if train_samples_per_epoch is None:
        train_samples_per_epoch = len(dataset.train_data) // seq_length

    train_ds = RandomWindowDataset(
        dataset.train_data, seq_length, train_samples_per_epoch
    )

    # Val/test: we want reproducible evaluation. The cleanest way is to
    # split the held-out data into non-overlapping windows and walk through
    # them sequentially. We implement this as a custom Dataset to avoid
    # randomness at evaluation time.
    val_ds  = SequentialWindowDataset(dataset.val_data,  seq_length)
    test_ds = SequentialWindowDataset(dataset.test_data, seq_length)

    # pin_memory speeds up GPU transfer by staging data in pinned RAM.
    # drop_last=True on training ensures every batch has the same shape,
    # which lets PyTorch optimize the LSTM's cuDNN backend.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,           # already random inside __getitem__
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sequential Window Dataset (for evaluation)
# ─────────────────────────────────────────────────────────────────────────────

class SequentialWindowDataset(Dataset):
    """
    Non-overlapping sequential windows for reproducible evaluation.

    We split the held-out split into chunks of (seq_length + 1) tokens
    (the +1 is for the target shift). Drop the last partial chunk if any.

    This is different from training, where random offsets give the model
    many overlapping views of the data. For evaluation, we want every
    token to be evaluated exactly once → non-overlapping windows.
    """

    def __init__(self, data: np.ndarray, seq_length: int):
        self.data = data
        self.seq_length = seq_length
        # Number of full (seq_length + 1) windows we can extract
        self.num_windows = (len(data) - 1) // seq_length

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_length
        end = start + self.seq_length
        x = self.data[start:end]
        y = self.data[start + 1 : end + 1]
        return (
            torch.from_numpy(x.copy()).long(),
            torch.from_numpy(y.copy()).long(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Raw Text Loader (convenience)
# ─────────────────────────────────────────────────────────────────────────────

def load_text_file(path: str) -> str:
    """
    Load a text file as a single string. Handles encoding gracefully.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_shakespeare() -> str:
    """
    Fetch Karpathy's tinyshakespeare (or read from cache).

    This is a great starter dataset for language modeling:
      - ~1 MB of text (fast iteration)
      - Coherent style (Shakespeare plays)
      - Manageable vocabulary (~10K unique words after min_freq=2)
      - Classic teaching example
    """
    url = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt"
    )
    cache = Path("data/raw/tinyshakespeare.txt")
    cache.parent.mkdir(parents=True, exist_ok=True)

    if not cache.exists():
        import urllib.request
        urllib.request.urlretrieve(url, cache)

    return cache.read_text(encoding='utf-8')


# ─────────────────────────────────────────────────────────────────────────────
# 7. Quick self-test (run this file directly to verify)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run `python src/data.py` to execute a quick smoke test of the pipeline.
    This catches bugs early without needing to run full training.
    """
    print("=" * 60)
    print("Running data.py smoke test")
    print("=" * 60)

    # 1. Load Shakespeare
    text = load_shakespeare()
    print(f"\nLoaded {len(text):,} characters of Shakespeare")

    # 2. Build dataset
    dataset = TextDataset(
        texts=[text],
        min_freq=2,
        val_ratio=0.05,
        test_ratio=0.05,
    )

    # 3. Save vocab
    vocab_path = Path("data/processed/vocab.pkl")
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_vocab(str(vocab_path))
    print(f"\nSaved vocab to {vocab_path}")

    # 4. Build loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        seq_length=32,
        batch_size=4,
    )

    # 5. Fetch one batch and verify shapes
    x, y = next(iter(train_loader))
    print(f"\nInput batch shape:  {x.shape}  (batch, seq_length)")
    print(f"Target batch shape: {y.shape}")
    print(f"Input dtype:  {x.dtype}")
    print(f"Target dtype: {y.dtype}")

    # 6. Verify the shift: y should be x shifted by 1 within each row
    assert torch.equal(x[:, 1:], y[:, :-1]), "Shift trick failed!"
    print("✓ Input/target shift verified")

    # 7. Round-trip: decode the first example
    first_x = x[0].tolist()
    first_y = y[0].tolist()
    decoded_x = dataset.vocab.decode(first_x)
    decoded_y = dataset.vocab.decode(first_y)
    print(f"\nDecoded input[0]:  {decoded_x[:80]}...")
    print(f"Decoded target[0]: {decoded_y[:80]}...")

    # 8. Vocab stats
    print(f"\nVocab size: {len(dataset.vocab):,}")
    print(f"Most common: {dataset.vocab.itos[4:14]}")

    print("\n" + "=" * 60)
    print("All smoke tests passed!")
    print("=" * 60)
