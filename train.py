"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device) -> Tensor
  evaluate_bleu(model, test_dataloader, tgt_vocab, device, max_len) -> float
  save_checkpoint(model, optimizer, scheduler, epoch, path) -> None
  load_checkpoint(path, model, optimizer, scheduler) -> int
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

import wandb
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask
from config import PAD_IDX, SOS_IDX, EOS_IDX, DEFAULT_CONFIG


class LabelSmoothingLoss(nn.Module):
    """
    Label smth as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth[correct] = 1 - eps
        y_smooth[others]  = eps / (v_sz - 2)   # exclude pad and correct
        y_smooth[pad]     = 0

    Uses KL divergence: KLDiv(log_softmax(logits), y_smooth).

    Args:
        v_sz : Number of output classes.
        p_idx    : Index of <pad> token (receives 0 probability).
        smth  : Smoothing factor eps (default 0.1).
    """

    def __init__(self, v_sz: int, p_idx: int, smth: float = 0.1) -> None:
        super().__init__()
        self.v_sz = v_sz
        self.p_idx    = p_idx
        self.smth  = smth
        self.conf = 1.0 - smth
        # Smooth mass distributed among all non-pad, non-correct tokens
        self.s_val = smth / max(1, v_sz - 2)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : [batch * tgt_len, v_sz]  (raw model output)
            target : [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        assert logits.size(0) == target.size(0)

        # Build smoothed distribution
        with torch.no_grad():
            s_dist = torch.full_like(logits, self.s_val)
            s_dist[:, self.p_idx] = 0.0          # pad gets 0
            s_dist.scatter_(1, target.unsqueeze(1), self.conf)  # correct gets conf
            # Zero out rows where the target itself is <pad>
            pad_mask = target.eq(self.p_idx)
            s_dist[pad_mask] = 0.0

        log_probs = F.log_softmax(logits, dim=-1)

        # KL divergence: sum( smooth * log_probs ), mean over non-pad tokens
        loss = -(s_dist * log_probs).sum(dim=-1)

        # Mask pad positions
        non_pad = (~pad_mask).float()
        loss = (loss * non_pad).sum() / non_pad.sum().clamp(min=1)
        return loss

def run_epoch(
    d_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    log_wandb: bool = True,
    grad_norm_log: bool = False,
    step_counter: Optional[list] = None,
    max_grad_log_steps: int = 1000,
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        d_iter       : DataLoader yielding (src, tgt) batches.
        model           : Transformer instance.
        loss_fn         : LabelSmoothingLoss (or any nn.Module loss).
        optimizer       : Optimizer (None during eval).
        scheduler       : NoamScheduler instance (None during eval).
        epoch_num       : Current epoch index (for logging).
        is_train        : If True, perform backward pass and scheduler step.
        device          : 'cpu' or 'cuda'.
        log_wandb       : Whether to log metrics to W&B.
        grad_norm_log   : If True, log Q/K gradient norms (ablation §2.2).
        step_counter    : Mutable list [int] tracking global step count.
        max_grad_log_steps: Stop grad-norm logging after this many steps.

    Returns:
        a_loss : Average loss over the epoch (float).
    """
    model.train() if is_train else model.eval()
    t_loss  = 0.0
    t_corr = 0
    t_toks = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    prefix  = "train" if is_train else "val"

    with context:
        pbar = tqdm(d_iter, desc=f"Epoch {epoch_num} [{prefix}]", leave=False)
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)

            # Teacher-forcing: decoder input is tgt[:-1], target is tgt[1:]
            t_inp  = tgt[:, :-1]
            t_tgt = tgt[:, 1:]

            src_mask = make_src_mask(src, PAD_IDX).to(device)
            tgt_mask = make_tgt_mask(t_inp, PAD_IDX).to(device)

            # Forward
            logits = model(src, t_inp, src_mask, tgt_mask)
            # logits: [B, tgt_len-1, v_sz]

            B, T, V = logits.size()
            loss = loss_fn(logits.contiguous().view(B * T, V),
                           t_tgt.contiguous().view(B * T))

            # Count non-pad tokens for averaging
            np_m = t_tgt.ne(PAD_IDX)
            n_toks = np_m.sum().item()
            t_loss   += loss.item() * n_toks
            t_toks += n_toks

            # Calculate accuracy
            preds = logits.argmax(dim=-1)
            correct = preds.eq(t_tgt)
            t_corr += (correct & np_m).sum().item()

            if is_train:
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Log Q/K gradient norms for ablation §2.2
                if grad_norm_log and step_counter is not None:
                    global_step = step_counter[0]
                    if global_step < max_grad_log_steps:
                        q_norms, k_norms = [], []
                        for layer in model.encoder.layers:
                            mha = layer.self_attn
                            if mha.W_q.weight.grad is not None:
                                q_norms.append(mha.W_q.weight.grad.norm().item())
                            if mha.W_k.weight.grad is not None:
                                k_norms.append(mha.W_k.weight.grad.norm().item())
                        if log_wandb and q_norms:
                            wandb.log({
                                "grad_norm/query": sum(q_norms) / len(q_norms),
                                "grad_norm/key":   sum(k_norms) / len(k_norms),
                                "global_step": global_step,
                            })
                    step_counter[0] += 1

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    a_loss = t_loss / max(1, t_toks)
    a_acc  = t_corr / max(1, t_toks)

    if log_wandb:
        wandb.log({
            f"{prefix}/loss": a_loss,
            f"{prefix}/accuracy": a_acc,
            "epoch": epoch_num,
        })

    return a_loss


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer (in eval mode).
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    with torch.no_grad():
        # Encode the source sequence once
        memory = model.encode(src, src_mask)  # [1, src_len, d_model]

        # Start with <sos>
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, PAD_IDX).to(device)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            # logits: [1, cur_len, v_sz]

            # Greedy: pick the highest-probability token at the last position
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            ys = torch.cat([ys, next_token], dim=1)

            if next_token.item() == end_symbol:
                break

    return ys

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (will be set to eval mode).
        test_dataloader : DataLoader over the test split.
        tgt_vocab       : Vocabulary with .lookup_token(idx) method.
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    try:
        import sacrebleu
        use_sacrebleu = True
    except ImportError:
        from nltk.translate.bleu_score import corpus_bleu
        use_sacrebleu = False

    model.eval()
    hypotheses = []
    references = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="Evaluating BLEU", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                single_src      = src[i].unsqueeze(0)
                single_src_mask = make_src_mask(single_src, PAD_IDX).to(device)

                pred_ids = greedy_decode(
                    model, single_src, single_src_mask,
                    max_len, SOS_IDX, EOS_IDX, device,
                )
                pred_ids = pred_ids.squeeze(0).tolist()

                # Remove <sos>/<eos>/<pad>
                special = {SOS_IDX, EOS_IDX, PAD_IDX}
                pred_tokens = [tgt_vocab.lookup_token(t) for t in pred_ids
                               if t not in special]
                ref_ids = tgt[i].tolist()
                ref_tokens  = [tgt_vocab.lookup_token(t) for t in ref_ids
                               if t not in special]

                hypotheses.append(" ".join(pred_tokens))
                references.append(" ".join(ref_tokens))

    if use_sacrebleu:
        result = sacrebleu.corpus_bleu(hypotheses, [references])
        return result.score
    else:
        # nltk expects list of list of tokens
        hyp_tok = [h.split() for h in hypotheses]
        ref_tok  = [[r.split()] for r in references]
        return corpus_bleu(ref_tok, hyp_tok) * 100.0


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimizer + scheduler state to disk.

    Saved dict keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    """
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config":         model.model_config,
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"]

def run_training_experiment(config: dict = None) -> None:
    """
    Set up and run the full training experiment with W&B logging.

    Steps:
        1. Init W&B
        2. Build dataset / vocabs
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer
        5. Instantiate Adam optimizer
        6. Instantiate NoamScheduler
        7. Instantiate LabelSmoothingLoss
        8. Training loop with best-model checkpointing
        9. Final BLEU on test set
    """
    import random
    import numpy as np
    from lr_scheduler import NoamScheduler
    from dataset import get_dataloaders

    if config is None:
        config = DEFAULT_CONFIG.copy()

    # ── Reproducibility ──────────────────────────────────────────────
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── W&B ─────────────────────────────────────────────────────────
    run = wandb.init(
        project="da6401-a3",
        config=config,
        name=config.get("run_name", "baseline"),
    )

    # ── Data ─────────────────────────────────────────────────────────
    print("Preparing data...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=config["batch_size"],
        max_src_len=config["max_src_len"],
        max_tgt_len=config["max_tgt_len"],
    )

    src_v_sz = len(src_vocab)
    tgt_v_sz = len(tgt_vocab)
    print(f"Src vocab: {src_v_sz}, Tgt vocab: {tgt_v_sz}")

    # ── Model ────────────────────────────────────────────────────────
    model = Transformer(
        src_v_sz=src_v_sz,
        tgt_v_sz=tgt_v_sz,
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        pe_mode=config.get("pe_mode", "sinusoidal"),
        use_attn_scale=config.get("use_attn_scale", True),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    wandb.log({"model/n_params": n_params})

    # ── Optimizer & Scheduler ────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["adam_lr"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_eps"],
    )

    use_noam = config.get("use_noam_scheduler", True)
    if use_noam:
        scheduler = NoamScheduler(
            optimizer,
            d_model=config["d_model"],
            warmup_steps=config["warmup_steps"],
        )
    else:
        # Fixed LR experiment (§2.1)
        fixed_lr = config.get("fixed_lr", 1e-4)
        for pg in optimizer.param_groups:
            pg["lr"] = fixed_lr
        scheduler = None

    # ── Loss ─────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        v_sz=tgt_v_sz,
        p_idx=PAD_IDX,
        smth=config.get("label_smth", 0.1),
    )

    # ── Training loop ────────────────────────────────────────────────
    best_val_loss = float("inf")
    checkpoint_path = config.get("checkpoint_path", "best_model.pt")

    for epoch in range(1, config["num_epochs"] + 1):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        # Log current LR
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"train/lr": current_lr, "epoch": epoch})

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={current_lr:.6f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler or
                            torch.optim.Adam(model.parameters()),
                            epoch, checkpoint_path)
            print(f"  => Saved best checkpoint (val_loss={val_loss:.4f})")

    # ── Final BLEU on test set ────────────────────────────────────────
    print("Computing test BLEU...")
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    wandb.log({"test/bleu": bleu})
    print(f"Test BLEU: {bleu:.2f}")

    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
