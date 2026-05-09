"""
config.py — Hyperparameter Configuration
DA6401 Assignment 3: "Attention Is All You Need"

Central place for all hyperparameters so experiments are reproducible
and easy to sweep via W&B.
"""

# ── Default training config (tuned for Multi30k) ──────────────────────
DEFAULT_CONFIG = dict(
    # Model architecture
    d_model      = 256,    # embedding / model dim  (paper: 512)
    N            = 3,      # encoder & decoder layers (paper: 6)
    num_heads    = 8,      # attention heads
    d_ff         = 512,    # feed-forward inner dim  (paper: 2048)
    dropout      = 0.1,

    # Training
    num_epochs   = 15,
    batch_size   = 128,
    label_smoothing = 0.1,
    clip_grad    = 1.0,    # gradient clipping max-norm

    # Optimiser (Adam from paper §5.3)
    adam_lr      = 1.0,    # base lr — Noam scheduler scales this
    adam_beta1   = 0.9,
    adam_beta2   = 0.98,
    adam_eps     = 1e-9,

    # Noam scheduler
    warmup_steps = 4000,

    # Data
    max_src_len  = 100,
    max_tgt_len  = 100,

    # Misc
    seed              = 42,
    checkpoint_path   = "best_model.pt",

    # Positional encoding mode: "sinusoidal" | "learned"
    pe_mode      = "sinusoidal",

    # Attention scaling (False → ablation without 1/√d_k)
    use_attn_scale = True,
)

# ── Paper-scale config (needs a strong GPU) ────────────────────────────
PAPER_CONFIG = DEFAULT_CONFIG.copy()
PAPER_CONFIG.update(
    d_model   = 512,
    N         = 6,
    d_ff      = 2048,
    num_epochs = 20,
    batch_size = 64,
)

# ── Special tokens ─────────────────────────────────────────────────────
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
