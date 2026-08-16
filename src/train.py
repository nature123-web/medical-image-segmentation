"""Train the U-Net.

    python -m src.train --config configs/base.yaml
    python -m src.train --config configs/base.yaml --loss dice_only
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_datasets, dataset_statistics
from .losses import build_loss
from .metrics import aggregate, evaluate_batch, format_report, remove_small_components
from .model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate_model(model, loader, device, threshold=0.5, tta=False,
                   min_component=0, compute_boundary=True):
    model.eval()
    all_results = []
    for images, masks in loader:
        probabilities = model.predict_proba(images.to(device), tta=tta)
        predictions = (probabilities >= threshold).float().cpu().numpy()[:, 0]
        targets = masks.numpy()[:, 0]
        if min_component > 0:
            predictions = np.stack([
                remove_small_components(p, min_component) for p in predictions
            ])
        all_results.extend(
            evaluate_batch(predictions, targets, compute_boundary)
        )
    return aggregate(all_results)


def tune_threshold(model, loader, device, candidates=None):
    """Pick the probability threshold maximising validation Dice.

    0.5 is only optimal when the classes are balanced. With a foreground
    occupying ~1% of the image the model is systematically under-confident, and
    the best operating point is usually well below 0.5 -- worth several Dice
    points for free.
    """
    candidates = candidates or np.arange(0.1, 0.91, 0.05)
    probabilities, targets = [], []
    with torch.no_grad():
        model.eval()
        for images, masks in loader:
            probabilities.append(
                model.predict_proba(images.to(device)).cpu().numpy()[:, 0]
            )
            targets.append(masks.numpy()[:, 0])
    probabilities = np.concatenate(probabilities)
    targets = np.concatenate(targets)

    best_threshold, best_dice = 0.5, -1.0
    for threshold in candidates:
        summary = aggregate(evaluate_batch(
            (probabilities >= threshold).astype(np.float32), targets,
            compute_boundary=False,
        ))
        if summary["dice"] > best_dice:
            best_threshold, best_dice = float(threshold), summary["dice"]
    return best_threshold, best_dice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--loss", default=None,
                        choices=["combined", "dice_only", "bce_only", "tversky"])
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.loss:
        presets = {
            "combined": dict(dice_weight=1.0, bce_weight=1.0, tversky_weight=0.0),
            "dice_only": dict(dice_weight=1.0, bce_weight=0.0, tversky_weight=0.0),
            "bce_only": dict(dice_weight=0.0, bce_weight=1.0, tversky_weight=0.0),
            "tversky": dict(dice_weight=0.0, bce_weight=0.5, tversky_weight=1.0),
        }
        cfg["loss"].update(presets[args.loss])

    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out_dir={out_dir}")

    train_ds, val_ds, test_ds = build_datasets(cfg)
    stats = dataset_statistics(train_ds, 200)
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    print(f"mean foreground: {stats['mean_foreground_fraction']:.3%}  "
          f"empty cases: {stats['empty_fraction']:.1%}")
    print("note: predicting all-background would score "
          f"{1 - stats['mean_foreground_fraction']:.2%} pixel accuracy "
          "and 0 Dice on lesion cases")

    loaders = {
        name: DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                         shuffle=(name == "train"),
                         num_workers=cfg["train"]["num_workers"])
        for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds))
    }

    model = build_model(cfg).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    criterion = build_loss(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    total_steps = cfg["train"]["epochs"] * max(1, len(loaders["train"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: cosine_with_warmup(s, cfg["train"]["warmup_steps"], total_steps),
    )

    target_boundary_weight = cfg["loss"].get("boundary_weight", 0.0)
    boundary_warmup = cfg["loss"].get("boundary_warmup_epochs", 0)

    history, best_dice, patience, best_state = [], -1.0, 0, None
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        # Ramp the boundary term in. Applied at full strength from step 0 it
        # dominates while the prediction is still far from the truth, and
        # training destabilises before Dice has found the lesion at all.
        if target_boundary_weight and boundary_warmup:
            ramp = min(1.0, (epoch - 1) / max(1, boundary_warmup))
            criterion.set_boundary_weight(target_boundary_weight * ramp)

        model.train()
        total, seen = 0.0, 0
        for images, masks in tqdm(loaders["train"], desc=f"epoch {epoch}",
                                  leave=False):
            images, masks = images.to(device), masks.to(device)
            loss = criterion(model(images), masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg["train"]["grad_clip"])
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * images.size(0)
            seen += images.size(0)

        summary = evaluate_model(model, loaders["val"], device,
                                 compute_boundary=False)
        print(f"epoch {epoch:3d}  loss {total/seen:.4f}  "
              f"val_Dice {summary['dice']:.4f}  "
              f"val_Dice(lesion) {summary['dice_lesion_only']:.4f}  "
              f"sens {summary['sensitivity']:.4f}")
        history.append({"epoch": epoch, "loss": total / seen, "val": summary})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2,
                                                         default=float))

        if summary["dice"] > best_dice:
            best_dice, patience = summary["dice"], 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["train"]["early_stopping_patience"]:
                print(f"early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    threshold, tuned_dice = tune_threshold(model, loaders["val"], device)
    print(f"\ntuned threshold: {threshold:.2f} (val Dice {tuned_dice:.4f}, "
          f"vs {best_dice:.4f} at 0.5)")

    print("\ntest set:")
    plain = evaluate_model(model, loaders["test"], device, threshold)
    print(format_report(plain, "U-Net"))

    with_tta = evaluate_model(model, loaders["test"], device, threshold, tta=True)
    print("\n" + format_report(with_tta, "U-Net + flip TTA"))

    cleaned = evaluate_model(model, loaders["test"], device, threshold, tta=True,
                             min_component=cfg["postprocess"]["min_component"])
    print("\n" + format_report(
        cleaned, f"U-Net + TTA + components >= "
                 f"{cfg['postprocess']['min_component']}px"))

    torch.save({"model": model.state_dict(), "config": cfg,
                "threshold": threshold}, out_dir / "best.pt")
    (out_dir / "test_metrics.json").write_text(json.dumps(
        {"plain": plain, "tta": with_tta, "tta_cleaned": cleaned,
         "threshold": threshold}, indent=2, default=float,
    ))
    print(f"\nsaved {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
