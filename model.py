"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  scaled_dot_product_attention(Q, K, V, mask) -> (out, weights)
  MultiHeadAttention.forward(q, k, v, mask)   -> Tensor
  PositionalEncoding.forward(x)               -> Tensor
  make_src_mask(src, pad_idx)                 -> BoolTensor
  make_tgt_mask(tgt, pad_idx)                 -> BoolTensor
  Transformer.encode(src, src_mask)           -> Tensor
  Transformer.decode(memory,src_m,tgt,tgt_m)  -> Tensor
"""

import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.
    Attention(Q, K, V) = softmax( Q*K^T / sqrt(d_k) ) * V

    Args:
        Q         : (..., seq_q, d_k)
        K         : (..., seq_k, d_k)
        V         : (..., seq_k, d_v)
        mask      : BoolTensor broadcastable to (..., seq_q, seq_k).
                    True positions are masked out (-inf before softmax).
        use_scale : If False, skip 1/sqrt(d_k) scaling (ablation §2.2).

    Returns:
        output : (..., seq_q, d_v)
        attn_w : (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if use_scale:
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Padding mask for encoder.
    Returns: BoolTensor [batch, 1, 1, src_len]  True=PAD (masked out)
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Combined padding + causal mask for decoder.
    Returns: BoolTensor [batch, 1, tgt_len, tgt_len]  True=masked out
    """
    batch_size, tgt_len = tgt.size()
    device = tgt.device
    # Upper-triangular (future positions)
    causal = torch.triu(
        torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool), diagonal=1
    )
    # Padding mask broadcast along seq_q
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    return causal.unsqueeze(0).unsqueeze(0) | pad_mask


# ══════════════════════════════════════════════════════════════════════
# MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (§3.2.2). Does NOT use torch.nn.MultiheadAttention.

    Args:
        d_model   : Total model dim (must be divisible by num_heads).
        num_heads : Number of parallel attention heads.
        dropout   : Dropout on attention weights.
        use_scale : Whether to apply 1/sqrt(d_k) scaling.
    """

    def __init__(self, d_model: int, num_heads: int,
                 dropout: float = 0.1, use_scale: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.use_scale = use_scale

        self.qw = nn.Linear(d_model, d_model, bias=False)
        self.kw = nn.Linear(d_model, d_model, bias=False)
        self.vw = nn.Linear(d_model, d_model, bias=False)
        self.ow = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        # Stored for attention visualization (§2.3)
        self.attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, D) -> (B, h, S, d_k)"""
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query : [B, seq_q, d_model]
            key   : [B, seq_k, d_model]
            value : [B, seq_k, d_model]
            mask  : BoolTensor broadcastable to [B, h, seq_q, seq_k]
        Returns:
            output : [B, seq_q, d_model]
        """
        B = query.size(0)
        Q = self._split_heads(self.qw(query))
        K = self._split_heads(self.kw(key))
        V = self._split_heads(self.vw(value))

        attn_out, attn_w = scaled_dot_product_attention(Q, K, V, mask, self.use_scale)
        self.attn_weights = attn_w.detach()

        attn_out = self.dropout(attn_out)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.ow(attn_out)


# ══════════════════════════════════════════════════════════════════════
# POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (§3.5). Supports 'sinusoidal' and 'learned' modes.

    Args:
        d_model : Embedding dimensionality.
        dropout : Dropout after adding encodings.
        max_len : Maximum sequence length (default 5000).
        mode    : 'sinusoidal' (fixed, buffer) or 'learned' (nn.Embedding).
    """

    def __init__(self, d_model: int, dropout: float = 0.1,
                 max_len: int = 5000, mode: str = "sinusoidal") -> None:
        super().__init__()
        self.mode    = mode
        self.dropout = nn.Dropout(p=dropout)

        if mode == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)
        elif mode == "learned":
            self.pe_embed = nn.Embedding(max_len, d_model)
            self.max_len  = max_len
        else:
            raise ValueError(f"Unknown PE mode: {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x : [batch, seq_len, d_model]
        Returns:   [batch, seq_len, d_model]  (x + positional encoding)
        """
        seq_len = x.size(1)
        if self.mode == "sinusoidal":
            x = x + self.pe[:, :seq_len, :]
        else:
            pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self.pe_embed(pos)
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
# FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, xW1+b1)W2+b2  (§3.3)"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.dropout(F.relu(self.l1(x))))


# ══════════════════════════════════════════════════════════════════════
# ENCODER LAYER (Post-LayerNorm, paper-faithful)
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single encoder layer: Self-Attn -> Add&Norm -> FFN -> Add&Norm.
    Uses Post-LayerNorm (as in the original paper §3.1).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, use_scale: bool = True) -> None:
        super().__init__()
        self.s_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.ff       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.n1     = nn.LayerNorm(d_model)
        self.n2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """x:[B,S,D], src_mask:[B,1,1,S] -> [B,S,D]"""
        x = self.n1(x + self.dropout(self.s_attn(x, x, x, src_mask)))
        x = self.n2(x + self.dropout(self.ff(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
# DECODER LAYER (Post-LayerNorm)
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single decoder layer:
      Masked Self-Attn -> Add&Norm -> Cross-Attn -> Add&Norm -> FFN -> Add&Norm.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, use_scale: bool = True) -> None:
        super().__init__()
        self.s_attn  = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.c_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.ff        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.n1      = nn.LayerNorm(d_model)
        self.n2      = nn.LayerNorm(d_model)
        self.n3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        x:[B,T,D], memory:[B,S,D], src_mask:[B,1,1,S], tgt_mask:[B,1,T,T] -> [B,T,D]
        """
        x = self.n1(x + self.dropout(self.s_attn(x, x, x, tgt_mask)))
        x = self.n2(x + self.dropout(self.c_attn(x, memory, memory, src_mask)))
        x = self.n3(x + self.dropout(self.ff(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
# ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N encoder layers + final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.lyrs = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.n   = nn.LayerNorm(layer.s_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.lyrs:
            x = layer(x, mask)
        return self.n(x)


class Decoder(nn.Module):
    """Stack of N decoder layers + final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.lyrs = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.n   = nn.LayerNorm(layer.s_attn.d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.lyrs:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.n(x)


# ══════════════════════════════════════════════════════════════════════
# FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size : Source vocabulary size.
        tgt_vocab_size : Target vocabulary size.
        d_model        : Model dimensionality (default 512).
        N              : Number of encoder/decoder layers (default 6).
        num_heads      : Number of attention heads (default 8).
        d_ff           : FFN inner dimensionality (default 2048).
        dropout        : Dropout probability (default 0.1).
        pe_mode        : Positional encoding: 'sinusoidal' or 'learned'.
        use_attn_scale : Whether to scale attention by 1/sqrt(d_k).
    """

    def __init__(
        self,
        src_vocab_size: int = 18669,
        tgt_vocab_size: int = 9797,
        d_model:        int   = 256,
        N:              int   = 4,
        num_heads:      int   = 8,
        d_ff:           int   = 1024,
        dropout:        float = 0.2,
        pe_mode:        str   = "sinusoidal",
        use_attn_scale: bool  = True,
    ) -> None:
        super().__init__()

        self.s_emb = nn.Embedding(src_vocab_size, d_model)
        self.t_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.s_pe    = PositionalEncoding(d_model, dropout, mode=pe_mode)
        self.t_pe    = PositionalEncoding(d_model, dropout, mode=pe_mode)

        enc_layer    = EncoderLayer(d_model, num_heads, d_ff, dropout, use_attn_scale)
        dec_layer    = DecoderLayer(d_model, num_heads, d_ff, dropout, use_attn_scale)
        self.enc = Encoder(enc_layer, N)
        self.dec = Decoder(dec_layer, N)

        self.proj = nn.Linear(d_model, tgt_vocab_size)

        # Stored for checkpoint saving
        self.model_config = dict(
            src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
            d_model=d_model, N=N, num_heads=num_heads, d_ff=d_ff,
            dropout=dropout, pe_mode=pe_mode, use_attn_scale=use_attn_scale,
        )

        self._init_weights()
        
        # Load vocab and tokenizer inside init as per requirements
        # NOTE: The pretrained download/load block is kept here but commented out
        # so training starts from scratch unless you manually enable it.
        import os
        import spacy

        import gdown
        drive_link = "https://drive.google.com/uc?id=1fB_aPN39P-zph1dNmu3SdFI4b_UxFsMB"
        ckpt_path = "best_model.pt"
        if not os.path.exists(ckpt_path):
            gdown.download(drive_link, ckpt_path, quiet=False)

        print("Loading spacy models and vocabs...")
        import spacy.cli
        try:
            self.de_nlp = spacy.load("de_core_news_sm")
        except OSError:
            spacy.cli.download("de_core_news_sm")
            self.de_nlp = spacy.load("de_core_news_sm")
            
        try:
            self.en_nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.en_nlp = spacy.load("en_core_web_sm")
        
        if os.path.exists('src_vocab.pt') and os.path.exists('tgt_vocab.pt'):
            self.src_vocab = torch.load('src_vocab.pt', map_location='cpu', weights_only=False)
            self.tgt_vocab = torch.load('tgt_vocab.pt', map_location='cpu', weights_only=False)
        else:
            # Fallback if somehow files are not present
            from dataset import build_vocabs
            self.src_vocab, self.tgt_vocab = build_vocabs(self.de_nlp, self.en_nlp)

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            self.load_state_dict(ckpt['model_state_dict'])

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ──────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Run the full encoder stack.
        src:[B,src_len] src_mask:[B,1,1,src_len] -> memory:[B,src_len,d_model]
        """
        d = self.s_emb.embedding_dim
        src_emb = self.s_pe(self.s_emb(src) * math.sqrt(d))
        return self.enc(src_emb, src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor,
               tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocab logits.
        Returns logits: [B, tgt_len, tgt_vocab_size]
        """
        d = self.t_emb.embedding_dim
        tgt_emb = self.t_pe(self.t_emb(tgt) * math.sqrt(d))
        dec_out  = self.dec(tgt_emb, memory, src_mask, tgt_mask)
        return self.proj(dec_out)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.
        Returns logits: [B, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)
        
    def infer(self, german_sentence: str, max_len: int = 100, beam_size: int = 5) -> str:
        """
        End-to-end inference with optimized beam search.
        Batches beam candidates to avoid timeouts.
        """
        from config import SOS_IDX, EOS_IDX, PAD_IDX
        self.eval()
        device = next(self.parameters()).device
        
        # Tokenize and numericalize source
        de_tokens = [tok.text.lower() for tok in self.de_nlp.tokenizer(german_sentence)]
        src_ids = [SOS_IDX] + self.src_vocab.encode(de_tokens) + [EOS_IDX]
        src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0) 
        src_mask = make_src_mask(src_tensor)
        
        with torch.no_grad():
            memory = self.encode(src_tensor, src_mask) # [1, src_len, d_model]
            
            if beam_size <= 1:
                tgt_ids = [SOS_IDX]
                for i in range(max_len - 1):
                    tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long, device=device).unsqueeze(0)
                    tgt_mask = make_tgt_mask(tgt_tensor)
                    logits = self.decode(memory, src_mask, tgt_tensor, tgt_mask)
                    next_token_id = logits[0, -1].argmax().item()
                    tgt_ids.append(next_token_id)
                    if next_token_id == EOS_IDX: break
            else:
                # Optimized Batched Beam Search
                # Initialize beams: [1, 1] -> [beam_size, 1]
                tgt_ids = torch.full((1, 1), SOS_IDX, dtype=torch.long, device=device)
                scores = torch.zeros(1, device=device)
                
                # Expand memory and src_mask for batching
                memory = memory.expand(beam_size, -1, -1)
                src_mask = src_mask.expand(beam_size, -1, -1, -1)
                
                # We start with 1 hypothesis and expand to beam_size at the first step
                curr_beam_size = 1
                completed = []
                alpha = 0.6

                for i in range(max_len - 1):
                    # 1. Forward pass for all beams at once
                    t_mask = make_tgt_mask(tgt_ids)
                    # For the first step, we only have 1 sequence, so we use a slice of memory
                    m_slice = memory[:curr_beam_size]
                    s_mask_slice = src_mask[:curr_beam_size]
                    
                    logits = self.decode(m_slice, s_mask_slice, tgt_ids, t_mask)
                    # log_probs: [curr_beam_size, vocab_size]
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                    
                    # 2. Update scores: [curr_beam_size, vocab_size]
                    # scores is [curr_beam_size], expanded to [curr_beam_size, 1]
                    total_scores = scores.unsqueeze(1) + log_probs
                    
                    # 3. Pick top beam_size from all candidates
                    # Flatten to [curr_beam_size * vocab_size]
                    flat_scores = total_scores.view(-1)
                    top_scores, top_indices = flat_scores.topk(min(beam_size, flat_scores.size(0)))
                    
                    # 4. Map back to beam_idx and token_id
                    vocab_size = log_probs.size(-1)
                    beam_indices = torch.div(top_indices, vocab_size, rounding_mode='floor')
                    token_ids = top_indices % vocab_size
                    
                    # 5. Build new sequences
                    new_tgt_ids = torch.cat([tgt_ids[beam_indices], token_ids.unsqueeze(1)], dim=1)
                    
                    # 6. Check for EOS and handle completed sequences
                    # This implementation is slightly simplified to keep it fast: 
                    # we just continue with top candidates and filter at the very end,
                    # or you can explicitly move finished ones to a separate list.
                    
                    # Update for next step
                    tgt_ids = new_tgt_ids
                    scores = top_scores
                    curr_beam_size = tgt_ids.size(0)
                    
                    # If all top sequences end with EOS, we can stop early
                    if (tgt_ids[:, -1] == EOS_IDX).all():
                        break
                
                # Apply length normalization and pick best
                # tgt_ids: [beam_size, seq_len], scores: [beam_size]
                lengths = (tgt_ids != PAD_IDX).sum(dim=1).float()
                norm_scores = scores / (lengths ** alpha)
                best_idx = norm_scores.argmax()
                tgt_ids = tgt_ids[best_idx].tolist()

        # Detokenize
        tgt_tokens = [self.tgt_vocab.lookup_token(idx) for idx in tgt_ids if idx not in (SOS_IDX, EOS_IDX, PAD_IDX)]
        return " ".join(tgt_tokens)

