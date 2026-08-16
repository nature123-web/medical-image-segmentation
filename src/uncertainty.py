"""Monte-Carlo dropout uncertainty for segmentation.

A binary mask tells a clinician *what* the model decided and nothing about how
sure it was. That is the wrong output for a tool meant to sit in front of a
radiologist: the useful behaviour is to segment confidently where it can and
flag the cases where it cannot, so review time goes to the ambiguous ones.

Keeping dropout active at inference and sampling several forward passes
approximates a posterior over the weights (Gal & Ghahramani, 2016). The spread
of those samples is the model's uncertainty.

The decomposition matters more than the total. Splitting uncertainty into

* **aleatoric** -- inherent ambiguity in the image; a genuinely fuzzy lesion
  boundary. More training data will not reduce it.
* **epistemic** -- the model's own ignorance; an unusual presentation it has not
  seen. More data *will* reduce it.

tells you which cases are worth labelling and which are simply hard.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def enable_dropout(model: nn.Module) -> int:
    """Put only the dropout layers into training mode. Returns how many.

    Calling ``model.train()`` would be wrong and is the usual mistake: it also
    reactivates BatchNorm's batch statistics, so predictions would depend on
    whatever else happened to be in the batch and the normalisation running
    estimates would drift. Only Dropout should be stochastic here.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()
            count += 1
    return count


def has_active_dropout(model: nn.Module) -> bool:
    """True if any dropout layer would actually drop anything."""
    return any(
        isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)) and m.p > 0
        for m in model.modules()
    )


def binary_entropy(probabilities: torch.Tensor, eps: float = 1e-7
                   ) -> torch.Tensor:
    """Shannon entropy of a Bernoulli, in nats. Maximal at p = 0.5."""
    p = probabilities.clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    images: torch.Tensor,
    n_samples: int = 20,
    warn_if_deterministic: bool = True,
) -> Dict[str, torch.Tensor]:
    """Sample the model with dropout active and summarise the spread.

    Returns a dict with:
        ``mean``       -- averaged probability, the prediction to threshold
        ``std``        -- per-pixel standard deviation across samples
        ``total``      -- predictive entropy of the mean
        ``aleatoric``  -- mean entropy of the individual samples
        ``epistemic``  -- total minus aleatoric (mutual information)

    All spatial maps have the same shape as the model output.
    """
    was_training = model.training
    model.eval()

    if warn_if_deterministic and not has_active_dropout(model):
        # Worth saying out loud: with dropout at 0 every sample is identical,
        # the variance is exactly 0, and the uncertainty map is meaningless
        # rather than merely small.
        print("warning: model has no active dropout (p=0); MC sampling will be "
              "deterministic and all uncertainty will read as zero. Train with "
              "model.dropout > 0 to use this.")

    enable_dropout(model)
    try:
        samples = []
        for _ in range(n_samples):
            logits = model(images)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            samples.append(torch.sigmoid(logits))
        stacked = torch.stack(samples)                  # (S, B, C, H, W)
    finally:
        model.train(was_training)

    mean = stacked.mean(dim=0)
    total = binary_entropy(mean)
    # Aleatoric is the entropy that remains even when the weights are known,
    # averaged over the posterior samples.
    aleatoric = binary_entropy(stacked).mean(dim=0)
    return {
        "mean": mean,
        "std": stacked.std(dim=0),
        "total": total,
        "aleatoric": aleatoric,
        # Non-negative in exact arithmetic; clamped because floating point can
        # produce a tiny negative when the two terms are nearly equal.
        "epistemic": (total - aleatoric).clamp_min(0.0),
        "samples": stacked,
    }


def case_uncertainty(uncertainty_map: torch.Tensor,
                     mask: torch.Tensor | None = None) -> float:
    """Collapse a per-pixel uncertainty map to one number per case.

    Averaged over the *predicted foreground and its surroundings* when a mask is
    supplied, rather than the whole image. Almost every pixel in a scan is
    confidently background, so a whole-image mean is dominated by easy
    background and barely moves between a clear case and a baffling one.
    """
    if mask is None:
        return float(uncertainty_map.mean())
    selection = mask.bool()
    if not selection.any():
        return float(uncertainty_map.mean())
    return float(uncertainty_map[selection].mean())


def rank_cases_for_review(uncertainties: np.ndarray, budget: float = 0.1
                          ) -> np.ndarray:
    """Indices of the most uncertain cases, up to a fraction of the cohort.

    The operational form of the whole idea: a reviewer has capacity for some
    percentage of cases, and these are the ones to spend it on.
    """
    uncertainties = np.asarray(uncertainties)
    n = max(1, int(len(uncertainties) * budget))
    return np.argsort(-uncertainties)[:n]


def uncertainty_error_correlation(uncertainties: np.ndarray,
                                  dice_scores: np.ndarray) -> float:
    """Spearman correlation between case uncertainty and error.

    The claim an uncertainty estimate has to earn: cases the model is unsure
    about should be the cases it gets wrong. A correlation near zero means the
    uncertainty is noise and routing review by it is no better than random.
    Returned as a *positive* number when uncertainty predicts low Dice.
    """
    from scipy import stats

    uncertainties = np.asarray(uncertainties, dtype=float)
    dice_scores = np.asarray(dice_scores, dtype=float)
    if len(uncertainties) < 3 or np.std(uncertainties) < 1e-12:
        return float("nan")
    return float(-stats.spearmanr(uncertainties, dice_scores).statistic)
