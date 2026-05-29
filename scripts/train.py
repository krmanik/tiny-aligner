"""
Training script for TinyAligner - Chinese CTC acoustic model.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_aligner.lexicon import merge_lexicons, build_phone_vocab
from tiny_aligner.dataset import get_dataloaders, set_specaugment_intensity
from tiny_aligner.model import build_model


DATA_ROOT        = "/home/mani/Documents/forced-aligner/data/aishell/data_aishell"
LEXICON_PATH     = "/home/mani/Documents/forced-aligner/data/aishell/resource_aishell/lexicon.txt"
CMUDICT_PATH     = Path(__file__).resolve().parent.parent / "data" / "lexicons" / "cmudict-0.7b"
LIBRISPEECH_ROOT = "/home/mani/Documents/forced-aligner/data"  # torchaudio appends LibriSpeech/ subdir
SAVE_DIR         = Path("checkpoints")


def ctc_decode_greedy(log_probs: torch.Tensor, blank_idx: int = 0) -> list:
    indices = log_probs.argmax(-1).tolist()
    decoded, prev = [], None
    for idx in indices:
        if idx != blank_idx and idx != prev:
            decoded.append(idx)
        prev = idx
    return decoded


def phone_error_rate(log_probs: torch.Tensor, targets: list, blank_idx: int = 0) -> float:
    hyp = ctc_decode_greedy(log_probs, blank_idx)
    ref = targets
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            new_dp[j] = min(new_dp[j - 1] + 1, dp[j] + 1, dp[j - 1] + cost)
        dp = new_dp
    return dp[m] / max(len(ref), 1)


def print_sample_predictions(model, dev_dl, device, blank_idx, idx2phone, n_samples=3):
    """Print decoded phonemes vs reference for a few dev samples.
    Use this to verify the model is learning real phonemes, not garbage."""
    model.eval()
    shown = 0
    sep = "-" * 64
    print(f"\n  {sep}")
    print(f"  SAMPLE DECODE CHECK  (greedy CTC decode vs reference)")
    print(f"  {sep}")

    with torch.no_grad():
        for batch in dev_dl:
            mel        = batch["mel"].to(device)
            phone_ids  = batch["phone_ids"]
            mel_lens   = batch["mel_lengths"].to(device)
            phone_lens = batch["phone_lengths"]
            texts      = batch["texts"]

            log_probs, out_lengths = model(mel, mel_lens)

            for i in range(min(n_samples - shown, mel.shape[0])):
                T   = out_lengths[i].item()
                lp  = log_probs[:T, i, :]
                ref = phone_ids[i, :phone_lens[i]].tolist()
                hyp = ctc_decode_greedy(lp, blank_idx)
                per = phone_error_rate(lp, ref, blank_idx)

                ref_ph = [idx2phone.get(r, "?") for r in ref]
                hyp_ph = [idx2phone.get(h, "?") for h in hyp]
                cap = 14
                ref_str = " ".join(ref_ph[:cap]) + (" ..." if len(ref_ph) > cap else "")
                hyp_str = " ".join(hyp_ph[:cap]) + (" ..." if len(hyp_ph) > cap else "")

                print(f"\n  [{shown+1}] text : {texts[i][:55]}")
                print(f"      REF  : {ref_str}")
                print(f"      HYP  : {hyp_str}")
                print(f"      PER  : {per:.3f}  ->  {(1-per)*100:.1f}% correct")
                shown += 1

            if shown >= n_samples:
                break

    print(f"  {sep}\n")
    model.train()


def train_epoch(model, loader, optimizer, scheduler,
                device, blank_idx, epoch, n_epochs,
                grad_clip=1.0, grad_spike_factor=50.0):
    """Train one epoch in fp32 (no AMP). CTC + AMP is numerically fragile
    and was producing late-training divergence on this dataset.

    Three guards protect against divergence:
      1. Skip step if loss is non-finite.
      2. Skip step if pre-clip grad norm exceeds grad_spike_factor * grad_clip
         (catches "huge but finite" grads that survive the isfinite check but
         poison weights even after clipping — direction still bad).
      3. Skip step if grad norm itself is non-finite.
    """
    model.train()
    ctc_loss_fn  = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    total_loss   = 0.0
    running_loss = 0.0
    accepted_steps = 0
    skipped_nan    = 0
    skipped_spike  = 0
    spike_threshold = grad_spike_factor * grad_clip
    log_interval = max(1, len(loader) // 10)

    pbar = tqdm(
        loader,
        desc=f"  Epoch {epoch:03d}/{n_epochs} [train]",
        unit="batch", ncols=90, file=sys.stdout, leave=False,
    )

    for step, batch in enumerate(pbar, 1):
        mel           = batch["mel"].to(device)
        phone_ids     = batch["phone_ids"].to(device)
        mel_lengths   = batch["mel_lengths"].to(device)
        phone_lengths = batch["phone_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)

        # fp32 throughout — no autocast, no GradScaler.
        log_probs, out_lengths = model(mel, mel_lengths)
        targets_flat = torch.cat([
            phone_ids[i, :phone_lengths[i]] for i in range(phone_ids.shape[0])
        ])
        loss = ctc_loss_fn(log_probs, targets_flat, out_lengths, phone_lengths)

        if not torch.isfinite(loss):
            skipped_nan += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        # Guard 2: huge but finite grad — skip entirely rather than clip-and-apply.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        gn = float(grad_norm)
        if not torch.isfinite(grad_norm):
            skipped_nan += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        if gn > spike_threshold:
            skipped_spike += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scheduler.step()

        loss_val       = loss.item()
        total_loss    += loss_val
        running_loss  += loss_val
        accepted_steps += 1
        lr_now         = scheduler.get_last_lr()[0]

        pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            avg=f"{total_loss/max(accepted_steps,1):.4f}",
            gn=f"{gn:.2f}",
            lr=f"{lr_now:.2e}",
        )

        if step % log_interval == 0:
            extras = ""
            if skipped_nan:   extras += f" | nan={skipped_nan}"
            if skipped_spike: extras += f" | spike={skipped_spike}"
            pbar.write(
                f"    step {step:5d}/{len(loader)} | "
                f"loss={loss_val:.4f} | "
                f"avg={running_loss/log_interval:.4f} | "
                f"gn={gn:.2f} | "
                f"lr={lr_now:.2e}" + extras
            )
            running_loss = 0.0

    pbar.close()
    if skipped_nan or skipped_spike:
        print(f"  [warn] skipped nan={skipped_nan} spike={skipped_spike} "
              f"(threshold {spike_threshold:.1f})")
    return total_loss / max(accepted_steps, 1)


def model_weights_healthy(model) -> bool:
    """Return False if any parameter has NaN or Inf."""
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return False
    return True


@torch.no_grad()
def evaluate(model, loader, device, blank_idx):
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    total_loss  = 0.0
    total_per   = 0.0
    n_utts      = 0

    pbar = tqdm(loader, desc="  [dev] evaluating", unit="batch",
                ncols=90, file=sys.stdout, leave=False)

    for batch in pbar:
        mel           = batch["mel"].to(device)
        phone_ids     = batch["phone_ids"].to(device)
        mel_lengths   = batch["mel_lengths"].to(device)
        phone_lengths = batch["phone_lengths"].to(device)

        log_probs, out_lengths = model(mel, mel_lengths)
        targets_flat = torch.cat([
            phone_ids[i, :phone_lengths[i]] for i in range(phone_ids.shape[0])
        ])
        loss = ctc_loss_fn(log_probs, targets_flat, out_lengths, phone_lengths)
        total_loss += loss.item()

        for i in range(mel.shape[0]):
            T   = out_lengths[i].item()
            lp  = log_probs[:T, i, :]
            ref = phone_ids[i, :phone_lengths[i]].tolist()
            total_per += phone_error_rate(lp, ref, blank_idx)
            n_utts += 1

        pbar.set_postfix(loss=f"{total_loss/len(loader):.4f}",
                         per=f"{total_per/max(n_utts,1):.4f}")

    pbar.close()
    return total_loss / len(loader), total_per / max(n_utts, 1)


def main():
    parser = argparse.ArgumentParser(description="Train TinyAligner")
    parser.add_argument("--epochs",       type=int,   default=80)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1.5e-3)
    parser.add_argument("--hidden",       type=int,   default=320)
    parser.add_argument("--rnn-layers",   type=int,   default=2,
                        help="Number of stacked Bi-GRU layers")
    parser.add_argument("--rnn-dropout",  type=float, default=0.1,
                        help="Dropout between stacked GRU layers")
    parser.add_argument("--patience",     type=int,   default=15)
    parser.add_argument("--specaug-decay-epochs", type=int, default=20,
                        help="Decay SpecAugment intensity 1.0 → --specaug-final "
                             "over this many final epochs (0 = no decay)")
    parser.add_argument("--specaug-final", type=float, default=0.4,
                        help="SpecAug intensity at the last epoch (0=off, 1=full)")
    parser.add_argument("--grad-clip",    type=float, default=1.0,
                        help="Max grad norm. Tighter (e.g. 0.5) if loss still spikes.")
    parser.add_argument("--no-balanced",  action="store_true",
                        help="Disable balanced ZH/EN sampling")
    parser.add_argument("--no-augment",   action="store_true",
                        help="Disable SpecAugment on training mels")
    parser.add_argument("--workers",      type=int,   default=4)
    parser.add_argument("--max-train",    type=int,   default=None,
                        help="Limit training samples (e.g. 5000 for quick test)")
    parser.add_argument("--resume",       type=str,   default=None)
    parser.add_argument("--sample-every", type=int,   default=5,
                        help="Show sample decode check every N epochs. 0 = off")
    parser.add_argument("--librispeech",  type=str,   default=LIBRISPEECH_ROOT,
                        help="Path to LibriSpeech root dir (default: LIBRISPEECH_ROOT)")
    parser.add_argument("--librispeech-splits", nargs="+",
                        default=["train-clean-100"],
                        help="LibriSpeech splits to use, e.g. train-clean-100 train-clean-360")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Lexicon & vocab (bilingual if CMUdict present, else Chinese-only)
    if CMUDICT_PATH.exists():
        print(f"[Train] Loading bilingual lexicon (AISHELL + CMUdict)...")
        lexicon = merge_lexicons(LEXICON_PATH, CMUDICT_PATH)
    else:
        from tiny_aligner.lexicon import load_lexicon
        print("[Train] Loading Chinese-only lexicon...")
        lexicon = load_lexicon(LEXICON_PATH)
    phone2idx, idx2phone = build_phone_vocab(lexicon)
    blank_idx = phone2idx["<blank>"]
    n_phones  = len(phone2idx)
    print(f"[Train] Phoneme vocab: {n_phones} tokens  (blank=0)")

    import json
    with open(save_dir /"phone2idx.json", "w", encoding="utf-8") as f:
        json.dump(phone2idx, f, ensure_ascii=False, indent=2)
    with open(save_dir /"idx2phone.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx2phone.items()}, f, ensure_ascii=False, indent=2)

    # Dataloaders
    print("[Train] Building dataloaders...")
    train_dl, dev_dl = get_dataloaders(
        DATA_ROOT, lexicon, phone2idx,
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_train_samples=args.max_train,
        librispeech_root=args.librispeech,
        librispeech_splits=args.librispeech_splits,
        balanced_sampling=not args.no_balanced,
        augment_train=not args.no_augment,
    )
    batches_per_epoch = len(train_dl)
    total_steps = args.epochs * batches_per_epoch
    print(f"[Train] {batches_per_epoch} batches/epoch x {args.epochs} epochs = {total_steps:,} total steps")
    print(f"[Train] ~{batches_per_epoch * args.batch_size} samples/epoch\n")

    # Model
    model = build_model(
        n_phones,
        hidden=args.hidden,
        num_rnn_layers=args.rnn_layers,
        rnn_dropout=args.rnn_dropout,
    ).to(device)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["model"])
        except RuntimeError as e:
            cfg = ckpt.get("config", {})
            print(f"[Train] Resume failed: checkpoint shape does not match current "
                  f"architecture.\n        ckpt config: {cfg}\n"
                  f"        current:     hidden={args.hidden} rnn_layers={args.rnn_layers}\n"
                  f"        Either match those flags or start fresh (omit --resume).\n"
                  f"        Underlying error: {e}")
            sys.exit(1)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[Train] Resumed from {args.resume}  (epoch {start_epoch})")

    # Optimizer & scheduler (fixed deprecation warnings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Do ONE warm-up step so OneCycleLR doesn't warn about step order at init
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr,
            total_steps=total_steps,
            pct_start=0.1, anneal_strategy="cos",
            last_epoch=-1,
        )
    # AMP disabled — pure fp32. CTC + autocast was causing late-training
    # divergence (clipped-but-direction-toxic grads at gn > 100 poisoned weights).

    best_per         = float("inf")
    patience         = args.patience
    patience_counter = 0
    ckpt             = None
    best_ckpt_path   = save_dir / "best_model.pt"

    print(f"[Train] lr={args.lr}  batch={args.batch_size}  hidden={args.hidden}  "
          f"rnn_layers={args.rnn_layers}  "
          f"clip={args.grad_clip}  patience={patience}  "
          f"sample-every={args.sample_every}  "
          f"balanced={not args.no_balanced}  augment={not args.no_augment}  "
          f"specaug_decay={args.specaug_decay_epochs}→{args.specaug_final}\n")

    def specaug_scale_for(epoch_idx: int) -> float:
        """Full intensity until last --specaug-decay-epochs, then linear ramp to
        --specaug-final over those final epochs. Disabled if --no-augment."""
        if args.no_augment or args.specaug_decay_epochs <= 0:
            return 1.0
        plateau_end = args.epochs - args.specaug_decay_epochs
        if epoch_idx <= plateau_end:
            return 1.0
        progress = (epoch_idx - plateau_end) / args.specaug_decay_epochs
        progress = min(max(progress, 0.0), 1.0)
        return 1.0 + (args.specaug_final - 1.0) * progress

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # SpecAugment curriculum — lighter masks in the final epochs so the
        # model fine-tunes on cleaner mels and squeezes out the last 1–2 PER pts.
        scale = specaug_scale_for(epoch)
        set_specaugment_intensity(scale)

        # Weight watchdog: if model has been poisoned with NaN/Inf weights,
        # reload best and continue. Catches the case where guards missed a
        # bad step that nuked weights mid-epoch.
        if not model_weights_healthy(model):
            print(f"[Train] Epoch {epoch}: model weights are non-finite. "
                  f"Reloading from {best_ckpt_path}.")
            if best_ckpt_path.exists():
                bckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(bckpt["model"])
            else:
                print(f"[Train] No best checkpoint to reload. Aborting.")
                break

        train_loss = train_epoch(
            model, train_dl, optimizer, scheduler,
            device, blank_idx, epoch, args.epochs,
            grad_clip=args.grad_clip,
        )

        dev_loss, dev_per = evaluate(model, dev_dl, device, blank_idx)
        elapsed  = time.time() - t0
        acc      = (1 - dev_per) * 100
        lr_now   = scheduler.get_last_lr()[0]
        improved = dev_per < best_per
        tag      = "NEW BEST" if improved else f"no improve {patience_counter+1}/{patience}"

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.4f} | "
            f"dev={dev_loss:.4f} | "
            f"PER={dev_per:.4f} | "
            f"Acc={acc:.1f}% | "
            f"lr={lr_now:.2e} | "
            f"specaug={scale:.2f} | "
            f"time={elapsed:.0f}s  [{tag}]"
        )

        # Sample decode check every N epochs
        if args.sample_every > 0 and epoch % args.sample_every == 0:
            print_sample_predictions(model, dev_dl, device, blank_idx, idx2phone, n_samples=3)

        ckpt = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "phone2idx": phone2idx,
            "idx2phone": idx2phone,
            "dev_per":   dev_per,
            "config": {
                "n_phones":      n_phones,
                "hidden":        args.hidden,
                "rnn_layers":    args.rnn_layers,
                "rnn_dropout":   args.rnn_dropout,
                "conv_channels": list(model.conv_channels),
            },
        }

        if improved:
            best_per         = dev_per
            patience_counter = 0
            torch.save(ckpt, save_dir /"best_model.pt")
            print(f"  Saved best_model.pt  (PER={best_per:.4f}  Acc={acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Train] Early stopping -- no improvement for {patience} epochs.")
                break

        torch.save(ckpt, save_dir /"latest_model.pt")
        print()   # blank line between epochs

    print(f"[Train] Done!  Best PER={best_per:.4f}  Acc={(1-best_per)*100:.1f}%")
    print(f"[Train] Best model -> {save_dir /'best_model.pt'}")


if __name__ == "__main__":
    main()
