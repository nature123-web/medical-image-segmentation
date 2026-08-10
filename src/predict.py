"""Segment images with a trained U-Net and render an overlay.

    python -m src.predict --checkpoint runs/base/best.pt --plot overlay.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import build_datasets
from .metrics import (
    aggregate,
    dice_score,
    evaluate_batch,
    format_report,
    remove_small_components,
)
from .model import build_model


def load_model(checkpoint: str | Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg, ckpt.get("threshold", 0.5)


@torch.no_grad()
def segment(model, images: torch.Tensor, device: torch.device,
            threshold: float = 0.5, tta: bool = True,
            min_component: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, binary masks) as numpy arrays."""
    probabilities = model.predict_proba(images.to(device), tta=tta)
    probabilities = probabilities.cpu().numpy()[:, 0]
    masks = (probabilities >= threshold).astype(np.float32)
    if min_component > 0:
        masks = np.stack([remove_small_components(m, min_component)
                          for m in masks])
    return probabilities, masks


def plot_overlays(images, targets, probabilities, masks, out_path, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n, len(images))
    fig, axes = plt.subplots(n, 4, figsize=(13, 3 * n), squeeze=False)
    for row in range(n):
        image = images[row, 0]
        axes[row][0].imshow(image, cmap="gray")
        axes[row][0].set_ylabel(f"case {row}")
        axes[row][1].imshow(targets[row], cmap="gray")
        axes[row][2].imshow(probabilities[row], cmap="magma", vmin=0, vmax=1)

        # Overlay: prediction in red, ground truth outline in green.
        axes[row][3].imshow(image, cmap="gray")
        overlay = np.zeros((*image.shape, 4))
        overlay[..., 0] = 1.0
        overlay[..., 3] = masks[row] * 0.45
        axes[row][3].imshow(overlay)
        axes[row][3].contour(targets[row], levels=[0.5], colors="lime",
                             linewidths=1.2)
        axes[row][3].set_title(
            f"Dice {dice_score(masks[row], targets[row]):.3f}", fontsize=9
        )
        for column, title in enumerate(
            ["image", "ground truth", "probability", "prediction"]
        ):
            if row == 0:
                axes[row][column].set_title(title, fontsize=10)
            axes[row][column].set_xticks([])
            axes[row][column].set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--plot", default=None)
    parser.add_argument("--n-examples", type=int, default=6)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg, tuned_threshold = load_model(args.checkpoint, device)
    threshold = args.threshold if args.threshold is not None else tuned_threshold
    print(f"threshold: {threshold:.2f}  TTA: {not args.no_tta}")

    _, _, test_ds = build_datasets(cfg)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=False)

    all_images, all_targets, all_probabilities, all_masks = [], [], [], []
    for images, targets in loader:
        probabilities, masks = segment(
            model, images, device, threshold, tta=not args.no_tta,
            min_component=cfg["postprocess"]["min_component"],
        )
        all_images.append(images.numpy())
        all_targets.append(targets.numpy()[:, 0])
        all_probabilities.append(probabilities)
        all_masks.append(masks)

    images = np.concatenate(all_images)
    targets = np.concatenate(all_targets)
    probabilities = np.concatenate(all_probabilities)
    masks = np.concatenate(all_masks)

    print("\n" + format_report(
        aggregate(evaluate_batch(masks, targets)), "U-Net"
    ))

    if args.plot:
        # Show the hardest cases, not the easiest -- a montage of successes
        # tells you nothing about where the model actually fails.
        order = np.argsort([dice_score(m, t) for m, t in zip(masks, targets)])
        pick = order[: args.n_examples]
        plot_overlays(images[pick], targets[pick], probabilities[pick],
                      masks[pick], args.plot, args.n_examples)


if __name__ == "__main__":
    main()
