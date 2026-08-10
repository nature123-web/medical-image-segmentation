"""Segmentation losses built for extreme class imbalance.

A lesion often occupies well under 1% of a scan. Plain binary cross-entropy on
that data is minimised by predicting background everywhere: the loss goes down,
pixel accuracy reads 99%+, and the Dice score is zero. Every loss here exists to
address some part of that failure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coefficient(probabilities: torch.Tensor, targets: torch.Tensor,
                     smooth: float = 1.0, per_sample: bool = True
                     ) -> torch.Tensor:
    """Soft Dice, differentiable in ``probabilities``.

    Computed **per sample** by default. Aggregating over the whole batch lets a
    single large structure dominate the denominator, so a small lesion the model
    missed entirely barely registers -- exactly the case that matters.
    """
    if per_sample:
        dims = tuple(range(1, probabilities.dim()))
    else:
        dims = tuple(range(probabilities.dim()))
    intersection = (probabilities * targets).sum(dims)
    cardinality = probabilities.sum(dims) + targets.sum(dims)
    # The smoothing term also defines the empty-mask case: with no foreground in
    # either prediction or target the score is 1, which is the correct answer.
    return (2 * intersection + smooth) / (cardinality + smooth)


class DiceLoss(nn.Module):
    """1 - soft Dice."""

    def __init__(self, smooth: float = 1.0, per_sample: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.per_sample = per_sample

    def forward(self, logits: torch.Tensor, targets: torch.Tensor
                ) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        return 1 - dice_coefficient(
            probabilities, targets, self.smooth, self.per_sample
        ).mean()


class TverskyLoss(nn.Module):
    """Generalised Dice with independent false-positive/negative weights.

    ``alpha`` weights false positives, ``beta`` false negatives; at
    ``alpha = beta = 0.5`` it is exactly Dice. Raising ``beta`` buys recall at
    the cost of precision, which is usually the right trade in screening
    applications where a missed lesion costs far more than a flagged
    false positive a radiologist dismisses in two seconds.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor
                ) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        dims = tuple(range(1, probabilities.dim()))
        true_positive = (probabilities * targets).sum(dims)
        false_positive = (probabilities * (1 - targets)).sum(dims)
        false_negative = ((1 - probabilities) * targets).sum(dims)
        tversky = (true_positive + self.smooth) / (
            true_positive + self.alpha * false_positive
            + self.beta * false_negative + self.smooth
        )
        return 1 - tversky.mean()


class FocalLoss(nn.Module):
    """Cross-entropy down-weighted on easy examples.

    The ``(1 - p_t) ** gamma`` factor shrinks the contribution of confidently
    classified pixels. On a scan that is 99% obvious background, this stops the
    sheer number of easy pixels from drowning out the few hundred that carry the
    signal.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor
                ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets,
                                                 reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class CombinedLoss(nn.Module):
    """Weighted sum of a region loss and a pixel loss.

    Dice alone has unstable gradients when the prediction is nearly empty (the
    denominator collapses) and gives no signal about calibration. BCE alone
    ignores the region structure. The sum is the standard, and it is standard
    because each covers the other's failure mode.
    """

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0,
                 focal_weight: float = 0.0, tversky_weight: float = 0.0,
                 pos_weight: float | None = None,
                 tversky_alpha: float = 0.3, tversky_beta: float = 0.7) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss(tversky_alpha, tversky_beta)
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight) if pos_weight is not None else None,
        )

    def forward(self, logits, targets) -> torch.Tensor:
        # Deep supervision returns a list; auxiliary heads are at lower
        # resolution, so the target is downsampled to meet them.
        if isinstance(logits, list):
            total = self._single(logits[0], targets)
            for index, auxiliary in enumerate(logits[1:], start=1):
                downsampled = F.interpolate(
                    targets, size=auxiliary.shape[-2:], mode="nearest"
                )
                total = total + 0.5 ** index * self._single(auxiliary, downsampled)
            return total
        return self._single(logits, targets)

    def _single(self, logits: torch.Tensor, targets: torch.Tensor
                ) -> torch.Tensor:
        total = logits.new_zeros(())
        if self.dice_weight:
            total = total + self.dice_weight * self.dice(logits, targets)
        if self.bce_weight:
            total = total + self.bce_weight * F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight
            )
        if self.focal_weight:
            total = total + self.focal_weight * self.focal(logits, targets)
        if self.tversky_weight:
            total = total + self.tversky_weight * self.tversky(logits, targets)
        return total


def build_loss(cfg: dict) -> CombinedLoss:
    loss_cfg = cfg["loss"]
    return CombinedLoss(
        dice_weight=loss_cfg["dice_weight"],
        bce_weight=loss_cfg["bce_weight"],
        focal_weight=loss_cfg["focal_weight"],
        tversky_weight=loss_cfg["tversky_weight"],
        pos_weight=loss_cfg["pos_weight"],
        tversky_alpha=loss_cfg["tversky_alpha"],
        tversky_beta=loss_cfg["tversky_beta"],
    )
