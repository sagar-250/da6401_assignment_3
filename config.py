"""
config.py — Hyperparameter Configuration
DA6401 Assignment 3: "Attention Is All You Need"

Central place for all hyperparameters so experiments are reproducible
and easy to sweep via W&B.
"""

# ── Default training config (tuned for Multi30k) ──────────────────────
DEFAULT_CONFIG = dict(
    d_model         = 256,
    N               = 4,
    num_heads       = 8,
    d_ff            = 1024,
    dropout         = 0.18,

    num_epochs      = 40,
    batch_size      = 128,

    label_smoothing = 0.11,
    clip_grad       = 1.0,

    adam_lr         = 1.0,
    adam_beta1      = 0.9,
    adam_beta2      = 0.98,
    adam_eps        = 5e-10,
    seed              = 42,
    checkpoint_path   = "best_model.pt",
    warmup_steps    = 4000,
    max_src_len  = 100,
    max_tgt_len  = 100,
    pe_mode         = "sinusoidal",
    use_attn_scale  = True,
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
